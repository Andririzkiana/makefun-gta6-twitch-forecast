# GTA VI Twitch CCU Forecast — a launch-day shape fitted from past launches

![make.fun bounty](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)
![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Andririzkiana/makefun-gta6-twitch-forecast/rank.svg)
![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Andririzkiana/makefun-gta6-twitch-forecast/reward.svg)

A Python program entered in make.fun's ["Forecast GTA VI's launch-day Twitch viewers hour by hour"](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) bounty on [make.fun](https://make.fun). Given a case file holding everything recorded up to the end of hour 1 of a release day, it writes five quantiles (5%, 25%, 50%, 75%, 95%) of the Twitch category's concurrent viewers for each of hours 2 to 24, scored by pinball loss on the log scale.

## The model

Nothing is memorised for one game; the forecast is fitted at run time from the case's own `comparables` — past big launches with full hourly curves:

- **Anchor.** Every forecast is hour 1's print times a ratio.
- **Shape.** Each comparable contributes its release day's log ratios ln(v_h / v_1): the shape a launch day takes after its first hour, as the unlock rolls through the world's time zones.
- **Similarity weighting.** A comparable counts more the closer its own hour-1 audience is to this case's (Gaussian kernel on the log hour-1 distance, bandwidth 1.1): big launches behave like big launches. Comparables that also launched on PC count 0.6, because every graded case here is a console-only launch and a PC release's simultaneous global unlock bends the day's shape.
- **Centre and spread.** Each hour's median is the weighted median of the log ratios, smoothed over neighbouring hours; the spread is two-sided — a weighted median deviation below the median and another above it, each rescaled to a sigma, floored, capped and smoothed — and the five quantiles are that log-scale distribution's normal quantiles.

Leave-one-out backtests over the 37 usable comparables in the kit (each launch forecast from the other 36, exactly as the program runs) score a mean pinball loss of **0.215** on the big rolling console launches (hour 1 ≥ 100k, pre-release evening at least 80% of hour 1), against 0.266 for the persistence baseline on the same launches; the baseline's published practice loss is 0.387.

## Layout

- `main.py` — the entrypoint (Python 3.13, standard library only). Reads every `*.in` in `$MAKEFUN_INPUT_DIR` (else `/input`), writes `<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (else `/output`). Deterministic: same case in, same bytes out; about 0.15 s a case.
- `LICENSE` — MIT.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
cat out/p00.out
```
