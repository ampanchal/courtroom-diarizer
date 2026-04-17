# compare_backends.py
import json

with open("outputs/pyannote_backend/freesound_community-2secondsection-59838/freesound_community-2secondsection-59838.json") as f:
    p = json.load(f)
with open("outputs/wavlm_backend/freesound_community-2secondsection-59838/freesound_community-2secondsection-59838.json") as f:
    w = json.load(f)

p_speakers = set(s["speaker"] for s in p["segments"])
w_speakers = set(s["speaker"] for s in w["segments"])

print("BACKEND COMPARISON")
print(f"pyannote : {len(p_speakers)} speakers  {len(p['segments'])} segments")
print(f"WavLM    : {len(w_speakers)} speakers  {len(w['segments'])} segments")
print()
print(f"{'Seg':<4} {'Start':>7} {'End':>7} {'Pyannote':<12} {'WavLM':<12}")
print("-" * 48)

for i in range(min(15, len(p["segments"]), len(w["segments"]))):
    ps = p["segments"][i]
    ws = w["segments"][i]
    print(f"{i+1:<4} {ps['start']:>7.2f} {ps['end']:>7.2f} "
          f"{ps['speaker']:<12} {ws['speaker']:<12}")