# -*- coding: utf-8 -*-
"""Phase 1: extract analysis audio, score audio quality, find multicam sync offset.
Usage: python analyze.py <ffmpeg> <ffprobe> <workdir> <pov1.mp4> <pov2.mp4> ..."""
import sys, os, json, subprocess
import numpy as np
import soundfile as sf
import librosa

FFMPEG, FFPROBE, WORK = sys.argv[1], sys.argv[2], sys.argv[3]
SRCS = sys.argv[4:]
SR = 22050  # analysis rate

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.returncode, p.stdout.decode('utf-8','ignore'), p.stderr.decode('utf-8','ignore')

def probe_dur(path):
    rc,o,e = run([FFPROBE,'-v','error','-show_entries','format=duration','-of','csv=p=0',path])
    try: return float(o.strip())
    except: return None

def extract(path, wav):
    if not os.path.exists(wav):
        run([FFMPEG,'-y','-i',path,'-vn','-ac','1','-ar',str(SR),wav])
    x,_ = sf.read(wav)
    return x.astype(np.float32)

def db(x): return 20*np.log10(max(x,1e-12))

def quality(x):
    # short-time RMS
    frame=2048; hop=1024
    n=1+(len(x)-frame)//hop
    rms=np.array([np.sqrt(np.mean(x[i*hop:i*hop+frame]**2)) for i in range(max(n,1))])
    rms=rms[rms>0]
    peak=float(np.max(np.abs(x)))
    overall_rms=float(np.sqrt(np.mean(x**2)))
    clip=float(np.mean(np.abs(x)>0.985))
    # noise floor = low percentile of frame energy; signal = high percentile
    noise=np.percentile(rms,5); sig=np.percentile(rms,75)
    snr=db(sig)-db(noise)
    crest=db(peak)-db(overall_rms)
    # spectral: brightness + balance
    S=np.abs(librosa.stft(x,n_fft=2048,hop_length=1024))
    freqs=librosa.fft_frequencies(sr=SR,n_fft=2048)
    psd=np.mean(S**2,axis=1)+1e-12
    cent=float(np.sum(freqs*psd)/np.sum(psd))
    hi=float(np.sum(psd[freqs>5000])/np.sum(psd))      # presence/air
    lo=float(np.sum(psd[freqs<200])/np.sum(psd))       # rumble/boom
    flat=float(np.exp(np.mean(np.log(psd)))/np.mean(psd))  # spectral flatness 0..1
    return dict(rms_db=round(db(overall_rms),2), peak_db=round(db(peak),2),
                clip_pct=round(clip*100,4), snr_db=round(float(snr),2),
                crest_db=round(float(crest),2), centroid_hz=round(cent,1),
                hi_ratio=round(hi,4), lo_ratio=round(lo,4), flatness=round(flat,4))

print("== EXTRACT + QUALITY ==", flush=True)
data={}
for s in SRCS:
    base=os.path.splitext(os.path.basename(s))[0]
    wav=os.path.join(WORK, base+".ana.wav")
    dur=probe_dur(s)
    print("extracting", base, "dur=", dur, flush=True)
    x=extract(s,wav)
    q=quality(x)
    data[base]=dict(path=s, wav=wav, dur=dur, **q)
    print(base, json.dumps(q), flush=True)

# ---- composite "better audio" score (higher=better) ----
# Penalize clipping heavily, reward SNR, reward presence(air), penalize excessive boom & extreme flatness(noise).
def score(q):
    return (q['snr_db']*1.0
            - q['clip_pct']*3.0
            + q['hi_ratio']*40.0
            - q['lo_ratio']*15.0
            - q['flatness']*30.0)
for b in data: data[b]['score']=round(score(data[b]),2)
ranked=sorted(data, key=lambda b:data[b]['score'], reverse=True)
print("\n== RANKING (best audio first) ==", flush=True)
for b in ranked: print("  %-22s score=%6.2f  clip=%.3f%% snr=%.1f air=%.3f boom=%.3f" % (
    b, data[b]['score'], data[b]['clip_pct'], data[b]['snr_db'], data[b]['hi_ratio'], data[b]['lo_ratio']), flush=True)
print("PICK:", ranked[0], flush=True)

# ---- sync offset of others vs the master (best-audio) via onset-envelope xcorr ----
master=ranked[0]
print("\n== SYNC (vs master=%s) ==" % master, flush=True)
def onset_env(wav, t0, t1):
    x,_=sf.read(wav)
    x=x[int(t0*SR):int(t1*SR)].astype(np.float32)
    return librosa.onset.onset_strength(y=x, sr=SR, hop_length=512)
mdur=data[master]['dur'] or 600
# use a stable mid window
w0=min(max(mdur*0.4,60), mdur-150) if mdur>200 else mdur*0.3
w1=w0+90
menv=onset_env(data[master]['wav'], w0, w1)
sync={}
for b in data:
    if b==master:
        sync[b]=0.0; continue
    # search other over a wider window to allow start offset +-120s
    oenv=onset_env(data[b]['wav'], max(w0-120,0), w1+120)
    corr=np.correlate(oenv-oenv.mean(), menv-menv.mean(), mode='full')
    lag=np.argmax(corr)-(len(menv)-1)
    sec=lag*512/SR
    # other_time = master_time + offset ; account for the -120 search base shift
    offset = (max(w0-120,0)) - w0 + sec
    sync[b]=round(float(offset),3)
    print("  %-22s offset=%+.3fs (drummer_time = master_time + offset)" % (b, sync[b]), flush=True)

data['_meta']=dict(master=master, ranking=ranked, sync_offset=sync, sr=SR)
with open(os.path.join(WORK,'analysis.json'),'w') as f: json.dump(data,f,indent=2)
print("\nWROTE analysis.json", flush=True)
