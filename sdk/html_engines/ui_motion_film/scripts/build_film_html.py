#!/usr/bin/env python3
import os

os.makedirs("artifacts/file_generation/ttl=63d/intermediate", exist_ok=True)
os.makedirs("artifacts/file_generation/ttl=63d/output", exist_ok=True)

HTML_CONTENT = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>10s 2D UI Motion Film — Reference-Accurate Deterministic Engine</title>
  <style>
    :root {
      --bg: #0D0D0F;
      --accent: #0A84FF;
      --text: #FFFFFF;
    }
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      user-select: none;
      -webkit-font-smoothing: antialiased;
    }
    html, body {
      width: 100%;
      height: 100%;
      background: #08080A;
      color: var(--text);
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
    }
    #viewport-wrap {
      position: relative;
      width: 100vw;
      height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #08080A;
    }
    #stage {
      width: 100%;
      height: 100%;
      max-width: calc(100vh * (16 / 9));
      max-height: calc(100vw * (9 / 16));
      aspect-ratio: 16 / 9;
      display: block;
      background: #0D0D0F;
      box-shadow: 0 24px 80px rgba(0, 0, 0, 0.8);
    }
    body.clean-mode #viewport-wrap {
      width: 1920px;
      height: 1080px;
    }
    body.clean-mode #stage {
      width: 1920px;
      height: 1080px;
      max-width: none;
      max-height: none;
      box-shadow: none;
    }
    body.clean-mode #hud {
      display: none !important;
    }
    #hud {
      position: fixed;
      bottom: 18px;
      left: 50%;
      transform: translateX(-50%);
      width: min(94vw, 1080px);
      background: rgba(22, 22, 26, 0.92);
      border: 1px solid #2E2E36;
      border-radius: 14px;
      padding: 10px 16px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      z-index: 20;
    }
    .hud-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }
    .hud-controls {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .btn {
      background: #232329;
      color: #FFFFFF;
      border: 1px solid #363640;
      border-radius: 8px;
      padding: 6px 12px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .btn:hover {
      background: #2C2C34;
    }
    .btn.active {
      background: #0A84FF;
      border-color: #47A1FF;
      color: #FFFFFF;
    }
    .beat-pills {
      display: flex;
      align-items: center;
      gap: 6px;
      flex-wrap: wrap;
    }
    .pill-btn {
      background: #1A1A20;
      color: #A0A0AB;
      border: 1px solid #2C2C35;
      border-radius: 6px;
      padding: 4px 9px;
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
    }
    .pill-btn:hover, .pill-btn.active {
      color: #FFFFFF;
      border-color: #0A84FF;
      background: rgba(10, 132, 255, 0.18);
    }
    .hud-metrics {
      font-family: 'DejaVu Sans Mono', monospace;
      font-size: 12px;
      color: #9494A0;
      display: flex;
      align-items: center;
      gap: 14px;
    }
    .hud-metrics strong {
      color: #FFFFFF;
    }
    .scrubber-row {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    #scrubber {
      width: 100%;
      accent-color: #0A84FF;
      cursor: pointer;
      height: 6px;
    }
  </style>
</head>
<body>
  <div id="viewport-wrap">
    <canvas id="stage" width="1920" height="1080"></canvas>
  </div>

  <div id="hud">
    <div class="hud-top">
      <div class="hud-controls">
        <button id="btn-play" class="btn active" type="button">Pause</button>
        <button id="btn-prev" class="btn" type="button">-1f</button>
        <button id="btn-next" class="btn" type="button">+1f</button>
        <button id="btn-audio" class="btn" type="button">Audio: ON</button>
      </div>
      <div class="beat-pills">
        <button class="pill-btn" data-seek="0.20" type="button">Toggle (0.20s)</button>
        <button class="pill-btn" data-seek="1.60" type="button">Slider (1.60s)</button>
        <button class="pill-btn" data-seek="2.52" type="button">Cards Fall (2.52s)</button>
        <button class="pill-btn" data-seek="5.02" type="button">Keyboard (5.02s)</button>
        <button class="pill-btn" data-seek="7.05" type="button">Return Key (7.05s)</button>
        <button class="pill-btn" data-seek="8.30" type="button">Full World (8.30s)</button>
      </div>
      <div class="hud-metrics">
        <span>TIME <strong id="lbl-time">0.000s</strong></span>
        <span>FRAME <strong id="lbl-frame">000</strong>/600</span>
        <span>BEAT <strong id="lbl-beat">01</strong>/20</span>
      </div>
    </div>
    <div class="scrubber-row">
      <input id="scrubber" type="range" min="0" max="10" step="0.001" value="0" />
    </div>
  </div>

  <script>
    (function () {
      const params = new URLSearchParams(window.location.search);
      if (params.get('clean') === '1') {
        document.body.classList.add('clean-mode');
      }

      const canvas = document.getElementById('stage');
      const ctx = canvas.getContext('2d');

      const W = 1920;
      const H = 1080;
      const DURATION = 10.0;
      const FPS = 60;
      const BPM = 120;
      const BEAT_SEC = 60 / BPM; // 0.50s

      const COLORS = {
        bg: '#0D0D0F',
        dot: '#252529',
        switchOff: '#323338',
        accent: '#0A84FF',
        trackBg: '#323338',
        tooltipBg: '#2C2C31',
        cardBg: '#232328',
        cardBorder: '#323238',
        keyBg: '#2C2C31',
        keyLip: '#1A1A1E',
        keyBorder: '#3A3A40',
        replyBg: '#1C1C21',
        replyBorder: '#383842',
        white: '#FFFFFF',
        textSub: '#ADADB5',
        textMuted: '#787882'
      };

      const BALL_R = 30;

      const WORLD = {
        toggle: {
          cx: -550,
          cy: -240,
          w: 154,
          h: 78,
          leftX: -588,
          rightX: -512,
          y: -240
        },
        slider: {
          x0: -250,
          x1: 390,
          y: -240,
          trackH: 8,
          tickY: -194,
          numTicks: 11
        },
        cards: [
          {
            id: 0,
            title: 'Slider',
            body: 'Reached 100',
            icon: 'slider',
            cx: 428,
            cy: -10,
            w: 396,
            h: 92,
            appearT: 1.70,
            hitT: 2.50,
            fadeT: 2.76
          },
          {
            id: 1,
            title: 'Gravity',
            body: 'Something is falling',
            icon: 'gravity',
            cx: 428,
            cy: 100,
            w: 396,
            h: 92,
            appearT: 2.10,
            hitT: 3.00,
            fadeT: 3.26
          },
          {
            id: 2,
            title: 'Messages',
            body: 'What now?',
            icon: 'messages',
            cx: 428,
            cy: 210,
            w: 396,
            h: 92,
            appearT: 2.30,
            hitT: 3.50,
            fadeT: 3.76
          }
        ],
        reply: {
          cx: 75,
          cy: 218,
          pillW: 196,
          pillH: 56,
          fieldW: 240,
          fieldH: 56,
          sentCx: 145,
          sentCy: 142,
          sentW: 112,
          sentH: 50
        },
        keyboard: {
          row1Y: 365,
          row2Y: 445,
          keyW: 68,
          keyH: 68,
          pitch: 80,
          rKeyX: 428
        }
      };

      const KEY_MAP = {};
      const KEY_LIST = [];
      (function buildKeys() {
        const kb = WORLD.keyboard;
        const r1 = ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'];
        r1.forEach((label, idx) => {
          const x = kb.rKeyX + (idx - 3) * kb.pitch;
          const y = kb.row1Y;
          const kObj = { id: label.toLowerCase(), label, x, y, w: kb.keyW, h: kb.keyH, row: 1 };
          KEY_MAP[kObj.id] = kObj;
          KEY_LIST.push(kObj);
        });

        const r2 = ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'];
        r2.forEach((label, idx) => {
          const x = kb.rKeyX + (idx - 2.65) * kb.pitch;
          const y = kb.row2Y;
          const kObj = { id: label.toLowerCase(), label, x, y, w: kb.keyW, h: kb.keyH, row: 2 };
          KEY_MAP[kObj.id] = kObj;
          KEY_LIST.push(kObj);
        });

        // Stepped ISO Return Key on the right of P and L
        const pKey = KEY_MAP['p'];
        const topW = 124;
        const botW = 98;
        const retH = (kb.row2Y - kb.row1Y) + kb.keyH; // 148px
        const leftTopX = pKey.x + kb.keyW / 2 + 12;
        const rightX = leftTopX + topW;
        const retObj = {
          id: 'return',
          label: 'return',
          x: (leftTopX + rightX) / 2,
          y: (kb.row1Y + kb.row2Y) / 2,
          w: topW,
          botW: botW,
          h: retH,
          topH: kb.keyH,
          isReturn: true
        };
        KEY_MAP['return'] = retObj;
        KEY_LIST.push(retObj);
      })();

      const KEY_HITS = [
        { t: 4.00, id: 'r', char: 'r' },
        { t: 4.50, id: 'e', char: 'e' },
        { t: 5.00, id: 't', char: 't' },
        { t: 5.50, id: 'r', char: 'r' },
        { t: 6.00, id: 'y', char: 'y' },
        { t: 7.00, id: 'return', char: '' }
      ];

      function clamp(v, lo, hi) {
        return Math.max(lo, Math.min(hi, v));
      }
      function lerp(a, b, u) {
        return a + (b - a) * u;
      }
      function smoothstep(u) {
        const x = clamp(u, 0, 1);
        return x * x * (3 - 2 * x);
      }
      function smootherstep(u) {
        const x = clamp(u, 0, 1);
        return x * x * x * (x * (x * 6 - 15) + 10);
      }
      function easeOutCubic(u) {
        const x = clamp(u, 0, 1);
        return 1 - Math.pow(1 - x, 3);
      }
      function easeInQuad(u) {
        const x = clamp(u, 0, 1);
        return x * x;
      }
      function easeOutQuad(u) {
        const x = clamp(u, 0, 1);
        return 1 - (1 - x) * (1 - x);
      }

      function springStep(dt, omega0 = 26.0, zeta = 0.68) {
        if (dt <= 0) return 0;
        if (dt > 0.70) return 1;
        const wd = omega0 * Math.sqrt(1 - zeta * zeta);
        const env = Math.exp(-zeta * omega0 * dt);
        return 1 - env * (Math.cos(wd * dt) + (zeta * omega0 / wd) * Math.sin(wd * dt));
      }

      function springImpulse(dt, omega0 = 34.0, zeta = 0.66) {
        if (dt <= 0 || dt > 0.45) return 0;
        const wd = omega0 * Math.sqrt(1 - zeta * zeta);
        return Math.exp(-zeta * omega0 * dt) * Math.sin(wd * dt);
      }

      function bounceArc(x0, y0, x1, y1, u, apexLift) {
        const uc = clamp(u, 0, 1);
        const x = lerp(x0, x1, smoothstep(uc));
        const yLinear = lerp(y0, y1, uc);
        const yParabola = -4 * apexLift * uc * (1 - uc);
        return { x, y: yLinear + yParabola };
      }

      function reboundDropArc(x0, y0, x1, y1, u, reboundH) {
        const uc = clamp(u, 0, 1);
        const x = lerp(x0, x1, smoothstep(uc));
        const split = 0.46;
        const apexY = y0 - reboundH;
        let y;
        if (uc < split) {
          const upU = uc / split;
          y = lerp(y0, apexY, easeOutQuad(upU));
        } else {
          const downU = (uc - split) / (1 - split);
          y = lerp(apexY, y1, easeInQuad(downU));
        }
        return { x, y };
      }

      function evalBall(t) {
        const tg = WORLD.toggle;
        const sl = WORLD.slider;
        const c0 = WORLD.cards[0];
        const c1 = WORLD.cards[1];
        const c2 = WORLD.cards[2];

        const c0TopY = c0.cy - c0.h / 2 - BALL_R + 2;
        const c1TopY = c1.cy - c1.h / 2 - BALL_R + 2;
        const c2TopY = c2.cy - c2.h / 2 - BALL_R + 2;

        const keyTopY = WORLD.keyboard.row1Y - WORLD.keyboard.keyH / 2 - BALL_R + 2;
        const retKey = KEY_MAP['return'];
        const retTopY = retKey.y - retKey.h / 2 - BALL_R + 2;

        if (t < 0.10) {
          return { x: tg.leftX, y: tg.y, sx: 1, sy: 1, rot: 0 };
        }
        if (t < 0.18) {
          const u = (t - 0.10) / 0.08;
          const x = lerp(tg.leftX, tg.rightX, easeOutCubic(u));
          return { x, y: tg.y, sx: 1.12, sy: 0.89, rot: 0 };
        }
        if (t < 0.50) {
          const u = (t - 0.18) / 0.32;
          const p = bounceArc(tg.rightX, tg.y, sl.x0, sl.y, u, 115);
          return { x: p.x, y: p.y, sx: 1.04, sy: 0.96, rot: 0 };
        }
        if (t < 2.12) {
          const dtLand = t - 0.50;
          const landSquash = springImpulse(dtLand, 36, 0.62) * 0.36;
          const uRoll = clamp((t - 0.54) / (2.10 - 0.54), 0, 1);
          const prog = 1 - Math.pow(1 - uRoll, 1.85);
          const x = lerp(sl.x0, sl.x1, prog);
          const sx = 1 + landSquash;
          const sy = 1 / sx;
          return { x, y: sl.y, sx, sy, rot: 0 };
        }
        // 2.12s -> 2.50s: Rolls off the right tip of the slider axis and falls under gravity onto the top of Card 1!
        if (t < 2.50) {
          const u = (t - 2.12) / (2.50 - 2.12);
          const x = lerp(sl.x1, c0.cx, easeOutQuad(u));
          const y = lerp(sl.y, c0TopY, Math.pow(u, 2.15));
          const fallStretch = Math.sin(Math.PI * u) * 0.16;
          return { x, y, sx: 1 - fallStretch * 0.5, sy: 1 + fallStretch, rot: 0 };
        }
        // 2.50s -> 3.00s: Impacts top of Card 1 ("Slider"), bounces up, Card 1 fades out, falls onto Card 2 ("Gravity")
        if (t < 3.00) {
          const dt = t - 2.50;
          const u = dt / 0.50;
          const p = reboundDropArc(c0.cx, c0TopY + 8, c1.cx, c1TopY, u, 72);
          const imp = springImpulse(dt, 42, 0.65) * 0.34;
          const sx = 1 + imp;
          const sy = 1 / sx;
          return { x: p.x, y: p.y, sx, sy, rot: 0 };
        }
        // 3.00s -> 3.50s: Impacts top of Card 2 ("Gravity"), bounces up, Card 2 fades out, falls onto Card 3 ("Messages")
        if (t < 3.50) {
          const dt = t - 3.00;
          const u = dt / 0.50;
          const p = reboundDropArc(c1.cx, c1TopY + 8, c2.cx, c2TopY, u, 72);
          const imp = springImpulse(dt, 42, 0.65) * 0.34;
          const sx = 1 + imp;
          const sy = 1 / sx;
          return { x: p.x, y: p.y, sx, sy, rot: 0 };
        }
        // 3.50s -> 4.00s: Impacts top of Card 3 ("Messages"), bounces up, Card 3 fades out, falls onto 'R' key
        if (t < 4.00) {
          const dt = t - 3.50;
          const u = dt / 0.50;
          const kR = KEY_MAP['r'];
          const p = reboundDropArc(c2.cx, c2TopY + 8, kR.x, keyTopY, u, 68);
          const imp = springImpulse(dt, 42, 0.65) * 0.34;
          const sx = 1 + imp;
          const sy = 1 / sx;
          return { x: p.x, y: p.y, sx, sy, rot: 0 };
        }
        if (t < 4.50) {
          const dt = t - 4.00;
          const u = dt / 0.50;
          const p = bounceArc(KEY_MAP['r'].x, keyTopY + 8, KEY_MAP['e'].x, keyTopY, u, 118);
          const imp = springImpulse(dt, 42, 0.65) * 0.32;
          return { x: p.x, y: p.y, sx: 1 + imp, sy: 1 / (1 + imp), rot: 0 };
        }
        if (t < 5.00) {
          const dt = t - 4.50;
          const u = dt / 0.50;
          const p = bounceArc(KEY_MAP['e'].x, keyTopY + 8, KEY_MAP['t'].x, keyTopY, u, 126);
          const imp = springImpulse(dt, 42, 0.65) * 0.32;
          return { x: p.x, y: p.y, sx: 1 + imp, sy: 1 / (1 + imp), rot: 0 };
        }
        if (t < 5.50) {
          const dt = t - 5.00;
          const u = dt / 0.50;
          const p = bounceArc(KEY_MAP['t'].x, keyTopY + 8, KEY_MAP['r'].x, keyTopY, u, 118);
          const imp = springImpulse(dt, 42, 0.65) * 0.32;
          return { x: p.x, y: p.y, sx: 1 + imp, sy: 1 / (1 + imp), rot: 0 };
        }
        if (t < 6.00) {
          const dt = t - 5.50;
          const u = dt / 0.50;
          const p = bounceArc(KEY_MAP['r'].x, keyTopY + 8, KEY_MAP['y'].x, keyTopY, u, 118);
          const imp = springImpulse(dt, 42, 0.65) * 0.32;
          return { x: p.x, y: p.y, sx: 1 + imp, sy: 1 / (1 + imp), rot: 0 };
        }
        if (t < 7.00) {
          const dt = t - 6.00;
          const u = dt / 1.00;
          const p = bounceArc(KEY_MAP['y'].x, keyTopY + 8, retKey.x + 12, retTopY, u, 420);
          const imp = springImpulse(dt, 42, 0.65) * 0.34;
          const airStretch = dt > 0.15 ? Math.sin(Math.PI * u) * 0.14 : 0;
          const sx = (1 + imp) * (1 - airStretch * 0.4);
          const sy = (1 / (1 + imp)) * (1 + airStretch);
          return { x: p.x, y: p.y, sx, sy, rot: 0 };
        }
        if (t < 7.62) {
          const dt = t - 7.00;
          const imp = springImpulse(dt, 38, 0.64) * 0.34;
          const holdDepress = dt < 0.45 ? smoothstep(dt / 0.12) * 10 : 10 * (1 - smoothstep((dt - 0.45) / 0.17));
          return {
            x: retKey.x + 12,
            y: retTopY + holdDepress,
            sx: 1 + imp,
            sy: 1 / (1 + imp),
            rot: 0
          };
        }
        if (t < 8.80) {
          const u = (t - 7.62) / (8.80 - 7.62);
          const p = bounceArc(retKey.x + 12, retTopY, tg.rightX, tg.y, u, 380);
          const stretch = Math.sin(Math.PI * u) * 0.15;
          return { x: p.x, y: p.y, sx: 1 + stretch, sy: 1 - stretch * 0.45, rot: -0.55 };
        }
        const dtReset = t - 8.80;
        const slideU = springStep(dtReset, 24, 0.72);
        const x = lerp(tg.rightX, tg.leftX, slideU);
        const imp = springImpulse(dtReset, 34, 0.68) * 0.18;
        return { x, y: tg.y, sx: 1 + imp, sy: 1 / (1 + imp), rot: 0 };
      }

      const CAM_KEYFRAMES = [
        { t: 0.00, x: -415, y: -240, zoom: 2.15 },
        { t: 0.25, x: -390, y: -240, zoom: 2.15 },
        { t: 0.65, x: -135, y: -240, zoom: 2.12 },
        { t: 1.65, x: 260,  y: -235, zoom: 2.12 },
        { t: 2.25, x: 410,  y: -115, zoom: 2.02 },
        { t: 2.65, x: 415,  y: 25,   zoom: 1.88 },
        { t: 3.15, x: 410,  y: 125,  zoom: 1.88 },
        { t: 3.65, x: 395,  y: 225,  zoom: 1.85 },
        { t: 4.15, x: 285,  y: 295,  zoom: 1.72 },
        { t: 5.00, x: 295,  y: 295,  zoom: 1.72 },
        { t: 6.05, x: 340,  y: 295,  zoom: 1.72 },
        { t: 6.85, x: 760,  y: 320,  zoom: 1.72 },
        { t: 7.55, x: 760,  y: 320,  zoom: 1.72 },
        { t: 8.30, x: 185,  y: 55,   zoom: 0.72 },
        { t: 8.60, x: 85,   y: 15,   zoom: 0.75 },
        { t: 9.15, x: -415, y: -240, zoom: 2.15 },
        { t: 10.00, x: -415, y: -240, zoom: 2.15 }
      ];

      function evalCamera(t) {
        const ct = clamp(t, 0, DURATION);
        for (let i = 0; i < CAM_KEYFRAMES.length - 1; i++) {
          const k0 = CAM_KEYFRAMES[i];
          const k1 = CAM_KEYFRAMES[i + 1];
          if (ct >= k0.t && ct <= k1.t) {
            const span = k1.t - k0.t;
            const u = span > 0 ? smootherstep((ct - k0.t) / span) : 0;
            const x = lerp(k0.x, k1.x, u);
            const y = lerp(k0.y, k1.y, u);
            const logZ = lerp(Math.log(k0.zoom), Math.log(k1.zoom), u);
            return { x, y, zoom: Math.exp(logZ) };
          }
        }
        const last = CAM_KEYFRAMES[CAM_KEYFRAMES.length - 1];
        return { x: last.x, y: last.y, zoom: last.zoom };
      }

      function roundRectPath(ctx, x, y, w, h, r) {
        const rad = Math.min(r, w / 2, h / 2);
        ctx.beginPath();
        ctx.moveTo(x + rad, y);
        ctx.lineTo(x + w - rad, y);
        ctx.quadraticCurveTo(x + w, y, x + w, y + rad);
        ctx.lineTo(x + w, y + h - rad);
        ctx.quadraticCurveTo(x + w, y + h, x + w - rad, y + h);
        ctx.lineTo(x + rad, y + h);
        ctx.quadraticCurveTo(x, y + h, x, y + h - rad);
        ctx.lineTo(x, y + rad);
        ctx.quadraticCurveTo(x, y, x + rad, y);
        ctx.closePath();
      }

      function fillRoundRect(ctx, x, y, w, h, r, fillStyle, strokeStyle, lineWidth = 1.5) {
        roundRectPath(ctx, x, y, w, h, r);
        if (fillStyle) {
          ctx.fillStyle = fillStyle;
          ctx.fill();
        }
        if (strokeStyle) {
          ctx.strokeStyle = strokeStyle;
          ctx.lineWidth = lineWidth;
          ctx.stroke();
        }
      }

      // Stepped ISO Return key path with rounded corners (matches exact_f420.png!)
      function isoReturnPath(ctx, x, y, topW, botW, h, topH, r) {
        const stepX = x + (topW - botW);
        const stepY = y + topH;
        const rightX = x + topW;
        const botY = y + h;
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.lineTo(rightX - r, y);
        ctx.quadraticCurveTo(rightX, y, rightX, y + r);
        ctx.lineTo(rightX, botY - r);
        ctx.quadraticCurveTo(rightX, botY, rightX - r, botY);
        ctx.lineTo(stepX + r, botY);
        ctx.quadraticCurveTo(stepX, botY, stepX, botY - r);
        ctx.lineTo(stepX, stepY + r);
        ctx.quadraticCurveTo(stepX, stepY, stepX - r, stepY);
        ctx.lineTo(x + r, stepY);
        ctx.quadraticCurveTo(x, stepY, x, stepY - r);
        ctx.lineTo(x, y + r);
        ctx.quadraticCurveTo(x, y, x + r, y);
        ctx.closePath();
      }

      function seek(rawTime) {
        let t = rawTime % DURATION;
        if (t < 0) t += DURATION;
        if (Math.abs(rawTime - DURATION) < 1e-6) t = DURATION;

        const cam = evalCamera(t);
        const ball = evalBall(t);

        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.fillStyle = COLORS.bg;
        ctx.fillRect(0, 0, W, H);

        ctx.save();
        ctx.translate(W / 2, H / 2);
        ctx.scale(cam.zoom, cam.zoom);
        ctx.translate(-cam.x, -cam.y);

        // 1. World-Space Faint Dot Grid
        const gridStep = 32;
        const viewHalfW = (W / 2) / cam.zoom + gridStep * 2;
        const viewHalfH = (H / 2) / cam.zoom + gridStep * 2;
        const startX = Math.floor((cam.x - viewHalfW) / gridStep) * gridStep;
        const endX = Math.ceil((cam.x + viewHalfW) / gridStep) * gridStep;
        const startY = Math.floor((cam.y - viewHalfH) / gridStep) * gridStep;
        const endY = Math.ceil((cam.y + viewHalfH) / gridStep) * gridStep;

        ctx.fillStyle = COLORS.dot;
        const dotR = 1.4;
        ctx.beginPath();
        for (let gx = startX; gx <= endX; gx += gridStep) {
          for (let gy = startY; gy <= endY; gy += gridStep) {
            ctx.moveTo(gx + dotR, gy);
            ctx.arc(gx, gy, dotR, 0, Math.PI * 2);
          }
        }
        ctx.fill();

        // 2. Frameless Toggle Switch
        (function drawToggle() {
          const tg = WORLD.toggle;
          let blueMix = 0;
          if (t >= 0.10 && t < 8.84) {
            blueMix = clamp((t - 0.10) / 0.08, 0, 1);
          } else if (t >= 8.84 && t < 8.96) {
            blueMix = 1 - clamp((t - 8.84) / 0.12, 0, 1);
          }

          const clickBounce = (t >= 0.08 && t < 0.30) ? springImpulse(t - 0.08, 36, 0.65) * (-0.06) : 0;
          const resetBounce = (t >= 8.80 && t < 9.20) ? springImpulse(t - 8.80, 32, 0.65) * 0.05 : 0;
          const sc = 1 + clickBounce + resetBounce;

          ctx.save();
          ctx.translate(tg.cx, tg.cy);
          ctx.scale(sc, sc);

          fillRoundRect(ctx, -tg.w / 2, -tg.h / 2, tg.w, tg.h, tg.h / 2, COLORS.switchOff, null, 0);
          if (blueMix > 0.001) {
            ctx.save();
            ctx.globalAlpha = blueMix;
            fillRoundRect(ctx, -tg.w / 2, -tg.h / 2, tg.w, tg.h, tg.h / 2, COLORS.accent, null, 0);
            ctx.restore();
          }

          ctx.restore();
        })();

        // 3. Frameless Horizontal Slider + Tick Marks + Odometer Value Bubble
        (function drawSlider() {
          const sl = WORLD.slider;
          const trackLen = sl.x1 - sl.x0;

          let prog = 0;
          if (t < 0.50) {
            prog = 0;
          } else if (t <= 2.12) {
            prog = clamp((ball.x - sl.x0) / trackLen, 0, 1);
          } else if (t <= 7.80) {
            prog = 1.0;
          } else if (t <= 8.48) {
            const rwU = (t - 7.80) / (8.48 - 7.80);
            prog = Math.pow(1 - clamp(rwU, 0, 1), 2.2);
          } else {
            prog = 0;
          }

          const fillX = lerp(sl.x0, sl.x1, prog);
          const exactVal = prog * 100;

          fillRoundRect(ctx, sl.x0, sl.y - sl.trackH / 2, trackLen, sl.trackH, sl.trackH / 2, COLORS.trackBg, null, 0);

          if (prog > 0.002) {
            const fw = Math.max(sl.trackH, fillX - sl.x0);
            fillRoundRect(ctx, sl.x0, sl.y - sl.trackH / 2, fw, sl.trackH, sl.trackH / 2, COLORS.accent, null, 0);
          }

          for (let i = 0; i < sl.numTicks; i++) {
            const tickFrac = i / (sl.numTicks - 1);
            const tx = lerp(sl.x0, sl.x1, tickFrac);
            const isBlue = prog > 0.005 && tickFrac <= prog + 0.005;
            fillRoundRect(
              ctx,
              tx - 2,
              sl.tickY - 7,
              4,
              14,
              2,
              isBlue ? COLORS.accent : COLORS.trackBg,
              null,
              0
            );
          }

          let bubbleScale = 0;
          if (t >= 0.52 && t <= 2.22) {
            const enterS = springStep(t - 0.52, 32, 0.68);
            const exitS = t > 2.12 ? 1 - smoothstep((t - 2.12) / 0.10) : 1;
            bubbleScale = enterS * exitS;
          } else if (t >= 7.80 && t <= 8.48) {
            const enterRw = smoothstep((t - 7.80) / 0.08);
            const exitRw = t > 8.38 ? 1 - smoothstep((t - 8.38) / 0.10) : 1;
            bubbleScale = enterRw * exitRw;
          }

          if (bubbleScale > 0.01) {
            const bx = fillX;
            const by = sl.y - 68;
            const bw = 64;
            const bh = 42;

            ctx.save();
            ctx.translate(bx, by);
            ctx.scale(bubbleScale, bubbleScale);

            fillRoundRect(ctx, -bw / 2, -bh / 2, bw, bh, 11, COLORS.tooltipBg, null, 0);

            ctx.fillStyle = COLORS.tooltipBg;
            ctx.beginPath();
            ctx.moveTo(-7, bh / 2 - 1);
            ctx.lineTo(7, bh / 2 - 1);
            ctx.lineTo(0, bh / 2 + 7);
            ctx.closePath();
            ctx.fill();

            ctx.save();
            roundRectPath(ctx, -bw / 2 + 2, -bh / 2 + 2, bw - 4, bh - 4, 9);
            ctx.clip();

            ctx.fillStyle = COLORS.white;
            ctx.font = '700 20px "Inter", sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';

            const intVal = Math.floor(exactVal);
            const frac = exactVal - intVal;
            if (exactVal >= 99.6) {
              ctx.fillText('100', 0, 1);
            } else if (frac > 0.15 && t < 2.08) {
              const tens = Math.floor(intVal / 10);
              const ones = intVal % 10;
              const nextOnes = (ones + 1) % 10;
              const prefix = tens > 0 ? String(tens) : '';
              const xOff = prefix ? 6 : 0;
              if (prefix) {
                ctx.fillText(prefix, -7, 1);
              }
              ctx.globalAlpha = 1 - frac * 0.6;
              ctx.fillText(String(ones), xOff, 1 - frac * 16);
              ctx.globalAlpha = frac * 0.85;
              ctx.fillText(String(nextOnes), xOff, 1 + (1 - frac) * 16);
            } else {
              ctx.fillText(String(Math.round(exactVal)), 0, 1);
            }

            ctx.restore();
            ctx.restore();
          }
        })();

        // 4. Three Notification Cards ("Slider", "Gravity", "Messages")
        WORLD.cards.forEach((card) => {
          let alpha = 0;
          let yOffset = 0;
          let scale = 1.0;

          if (t >= card.appearT && t <= card.fadeT + 0.18) {
            const enterU = clamp((t - card.appearT) / 0.18, 0, 1);
            alpha = smoothstep(enterU);
            yOffset = (1 - springStep(t - card.appearT, 26, 0.72)) * 40;

            if (t >= card.hitT) {
              const dtHit = t - card.hitT;
              yOffset += springImpulse(dtHit, 32, 0.65) * 16;
            }

            if (t > card.fadeT) {
              const fadeU = clamp((t - card.fadeT) / 0.16, 0, 1);
              alpha *= (1 - fadeU);
              yOffset -= fadeU * 18;
              scale = 1 - fadeU * 0.04;
            }
          } else if (t >= 7.92 && t <= 8.88) {
            const reEnter = clamp((t - (7.92 + card.id * 0.08)) / 0.20, 0, 1);
            const reExit = t > 8.68 ? 1 - clamp((t - 8.68) / 0.18, 0, 1) : 1;
            alpha = smoothstep(reEnter) * smoothstep(reExit);
            yOffset = (1 - smoothstep(reEnter)) * 24;
          }

          if (alpha <= 0.01) return;

          const cx = card.cx;
          const cy = card.cy + yOffset;
          const w = card.w;
          const h = card.h;
          const x = cx - w / 2;
          const y = cy - h / 2;

          ctx.save();
          ctx.globalAlpha = alpha;
          ctx.translate(cx, cy);
          ctx.scale(scale, scale);
          ctx.translate(-cx, -cy);

          fillRoundRect(ctx, x, y, w, h, 20, COLORS.cardBg, COLORS.cardBorder, 1.4);

          const icX = x + 20;
          const icY = y + (h - 46) / 2;
          fillRoundRect(ctx, icX, icY, 46, 46, 11, COLORS.accent, null, 0);

          ctx.strokeStyle = COLORS.white;
          ctx.fillStyle = COLORS.white;
          ctx.lineWidth = 2.8;
          ctx.lineCap = 'round';
          ctx.lineJoin = 'round';

          const icCx = icX + 23;
          const icCy = icY + 23;
          if (card.icon === 'slider') {
            ctx.beginPath();
            ctx.moveTo(icCx - 12, icCy);
            ctx.lineTo(icCx + 12, icCy);
            ctx.stroke();
            ctx.beginPath();
            ctx.arc(icCx + 3, icCy, 5.5, 0, Math.PI * 2);
            ctx.fill();
          } else if (card.icon === 'gravity') {
            ctx.beginPath();
            ctx.moveTo(icCx, icCy - 10);
            ctx.lineTo(icCx, icCy + 10);
            ctx.moveTo(icCx - 7, icCy + 3);
            ctx.lineTo(icCx, icCy + 10);
            ctx.lineTo(icCx + 7, icCy + 3);
            ctx.stroke();
          } else if (card.icon === 'messages') {
            ctx.beginPath();
            ctx.roundRect(icCx - 11, icCy - 9, 22, 16, 6);
            ctx.fill();
            ctx.beginPath();
            ctx.moveTo(icCx - 6, icCy + 5);
            ctx.lineTo(icCx - 10, icCy + 11);
            ctx.lineTo(icCx - 1, icCy + 6);
            ctx.closePath();
            ctx.fill();
          }

          ctx.fillStyle = COLORS.white;
          ctx.font = '700 19px "Inter", sans-serif';
          ctx.textAlign = 'left';
          ctx.textBaseline = 'middle';
          ctx.fillText(card.title, x + 82, y + 32);

          ctx.fillStyle = COLORS.textMuted;
          ctx.font = '500 16px "Inter", sans-serif';
          ctx.textAlign = 'right';
          ctx.fillText('now', x + w - 22, y + 32);

          ctx.fillStyle = COLORS.textSub;
          ctx.font = '500 18px "Inter", sans-serif';
          ctx.textAlign = 'left';
          ctx.fillText(card.body, x + 82, y + 61);

          ctx.restore();
        });

        // 5. Reply Pill / Input Field & Sent Blue Message
        (function drawReplyAndMessage() {
          const rp = WORLD.reply;

          if (t >= 2.55 && t <= 6.45) {
            const enterAlpha = clamp((t - 2.55) / 0.25, 0, 1);
            const w = lerp(rp.pillW, rp.fieldW, springStep(t - 3.90, 26, 0.70));
            const h = rp.fieldH;
            const x = rp.cx - w / 2;
            const y = rp.cy - h / 2;

            ctx.save();
            ctx.globalAlpha = enterAlpha;

            fillRoundRect(ctx, x, y, w, h, h / 2, COLORS.replyBg, COLORS.replyBorder, 1.6);

            if (t < 3.92) {
              ctx.fillStyle = COLORS.textMuted;
              ctx.font = '500 19px "Inter", sans-serif';
              ctx.textAlign = 'left';
              ctx.textBaseline = 'middle';
              ctx.fillText('Reply', x + 24, rp.cy + 1);
            } else {
              let typed = '';
              let latestCharProgress = 1;
              let latestChar = '';
              for (let i = 0; i < 5; i++) {
                const kh = KEY_HITS[i];
                if (t >= kh.t) {
                  const dtChar = t - kh.t;
                  if (dtChar < 0.12) {
                    latestChar = kh.char;
                    latestCharProgress = easeOutCubic(dtChar / 0.12);
                  } else {
                    typed += kh.char;
                  }
                }
              }

              const textX = x + 26;
              ctx.fillStyle = COLORS.white;
              ctx.font = '500 21px "Inter", sans-serif';
              ctx.textAlign = 'left';
              ctx.textBaseline = 'middle';
              ctx.fillText(typed, textX, rp.cy + 1);

              let caretX = textX + ctx.measureText(typed).width;
              if (latestChar) {
                ctx.save();
                ctx.globalAlpha = enterAlpha * latestCharProgress;
                ctx.fillText(latestChar, caretX, rp.cy + 1 + (1 - latestCharProgress) * 10);
                ctx.restore();
                caretX += ctx.measureText(latestChar).width;
              }

              fillRoundRect(ctx, caretX + 2, rp.cy - 13, 2.4, 26, 1.2, COLORS.accent, null, 0);

              const btnScale = springStep(t - 3.95, 30, 0.65);
              const btnX = x + w - 28;
              const btnY = rp.cy;
              ctx.save();
              ctx.translate(btnX, btnY);
              ctx.scale(btnScale, btnScale);
              ctx.fillStyle = COLORS.accent;
              ctx.beginPath();
              ctx.arc(0, 0, 17, 0, Math.PI * 2);
              ctx.fill();

              ctx.strokeStyle = COLORS.white;
              ctx.lineWidth = 2.4;
              ctx.lineCap = 'round';
              ctx.lineJoin = 'round';
              ctx.beginPath();
              ctx.moveTo(0, 7);
              ctx.lineTo(0, -7);
              ctx.moveTo(-6, -1);
              ctx.lineTo(0, -7);
              ctx.lineTo(6, -1);
              ctx.stroke();
              ctx.restore();
            }

            ctx.restore();
          }

          if (t >= 7.85 && t <= 8.85) {
            const wAlpha = smoothstep(clamp((t - 7.85) / 0.18, 0, 1)) * (t > 8.68 ? 1 - smoothstep((t - 8.68) / 0.17) : 1);
            if (wAlpha > 0.01) {
              ctx.save();
              ctx.globalAlpha = wAlpha;

              fillRoundRect(ctx, rp.cx - 85, rp.cy - 24, 170, 48, 24, COLORS.replyBg, COLORS.replyBorder, 1.5);
              ctx.fillStyle = COLORS.textMuted;
              ctx.font = '500 17px "Inter", sans-serif';
              ctx.textAlign = 'left';
              ctx.textBaseline = 'middle';
              ctx.fillText('Reply', rp.cx - 63, rp.cy + 1);

              fillRoundRect(
                ctx,
                rp.sentCx - rp.sentW / 2,
                rp.sentCy - rp.sentH / 2,
                rp.sentW,
                rp.sentH,
                rp.sentH / 2,
                COLORS.accent,
                null,
                0
              );
              ctx.fillStyle = COLORS.white;
              ctx.font = '600 18px "Inter", sans-serif';
              ctx.textAlign = 'center';
              ctx.textBaseline = 'middle';
              ctx.fillText('retry', rp.sentCx, rp.sentCy + 1);

              ctx.restore();
            }
          }
        })();

        // 6. Frameless Mechanical QWERTY Keyboard + Stepped ISO Return Key
        (function drawKeyboard() {
          if (t < 2.95 || t > 8.85) return;
          const enterAlpha = clamp((t - 2.95) / 0.30, 0, 1);
          const exitAlpha = t > 8.68 ? 1 - clamp((t - 8.68) / 0.17, 0, 1) : 1;
          const alpha = enterAlpha * exitAlpha;
          if (alpha <= 0.01) return;

          const riseY = (1 - springStep(t - 2.95, 22, 0.74)) * 70;

          ctx.save();
          ctx.globalAlpha = alpha;
          ctx.translate(0, riseY);

          KEY_LIST.forEach((kObj) => {
            let depressY = 0;
            for (let i = 0; i < KEY_HITS.length; i++) {
              const kh = KEY_HITS[i];
              if (kh.id === kObj.id && t >= kh.t) {
                const dt = t - kh.t;
                if (kObj.isReturn && dt < 0.62) {
                  const hold = dt < 0.45 ? smoothstep(dt / 0.10) * 11 : 11 * (1 - smoothstep((dt - 0.45) / 0.17));
                  depressY = Math.max(depressY, hold + springImpulse(dt, 34, 0.65) * 6);
                } else {
                  depressY = Math.max(depressY, springImpulse(dt, 36, 0.64) * 14);
                }
              }
            }

            const kx = kObj.x;
            const ky = kObj.y + depressY;
            const kw = kObj.w;
            const kh = kObj.h;
            const x = kx - kw / 2;
            const y = ky - kh / 2;

            if (kObj.isReturn) {
              // Stepped ISO Return Key
              isoReturnPath(ctx, x, y + 4, kw, kObj.botW, kh, kObj.topH, 11);
              ctx.fillStyle = COLORS.keyLip;
              ctx.fill();

              isoReturnPath(ctx, x, y, kw, kObj.botW, kh, kObj.topH, 11);
              ctx.fillStyle = COLORS.keyBg;
              ctx.fill();
              ctx.strokeStyle = COLORS.keyBorder;
              ctx.lineWidth = 1.2;
              ctx.stroke();

              // Return arrow ↵
              ctx.strokeStyle = COLORS.white;
              ctx.lineWidth = 2.6;
              ctx.lineCap = 'round';
              ctx.lineJoin = 'round';

              const arrowCx = kx + 10;
              const arrowCy = y + 36;
              ctx.beginPath();
              ctx.moveTo(arrowCx + 16, arrowCy - 10);
              ctx.lineTo(arrowCx + 16, arrowCy + 2);
              ctx.lineTo(arrowCx - 14, arrowCy + 2);
              ctx.moveTo(arrowCx - 7, arrowCy - 5);
              ctx.lineTo(arrowCx - 14, arrowCy + 2);
              ctx.lineTo(arrowCx - 7, arrowCy + 9);
              ctx.stroke();

              ctx.fillStyle = COLORS.textSub;
              ctx.font = '500 16px "Inter", sans-serif';
              ctx.textAlign = 'right';
              ctx.textBaseline = 'bottom';
              ctx.fillText('return', x + kw - 14, y + kh - 12);
            } else {
              fillRoundRect(ctx, x, y + 4, kw, kh, 11, COLORS.keyLip, null, 0);
              fillRoundRect(ctx, x, y, kw, kh, 11, COLORS.keyBg, COLORS.keyBorder, 1.2);

              ctx.fillStyle = COLORS.white;
              ctx.font = '500 22px "Inter", sans-serif';
              ctx.textAlign = 'center';
              ctx.textBaseline = 'middle';
              ctx.fillText(kObj.label, kx, ky + 1);
            }
          });

          ctx.restore();
        })();

        // 7. White Ball
        (function drawBall() {
          ctx.save();
          ctx.translate(ball.x, ball.y);
          if (ball.rot) ctx.rotate(ball.rot);
          ctx.scale(ball.sx, ball.sy);

          ctx.fillStyle = COLORS.white;
          ctx.beginPath();
          ctx.arc(0, 0, BALL_R, 0, Math.PI * 2);
          ctx.fill();

          ctx.restore();
        })();

        // 8. Physical Pointer Cursor
        (function drawCursor() {
          const tg = WORLD.toggle;
          const hoverX = tg.leftX + 24;
          const hoverY = tg.y + 16;
          const parkedX = tg.leftX + 5;
          const parkedY = tg.y + 128;

          let cx = hoverX;
          let cy = hoverY;
          let cScale = 1.0;

          if (t < 0.08) {
            cx = hoverX;
            cy = hoverY;
          } else if (t < 0.22) {
            const pU = Math.sin(Math.PI * clamp((t - 0.08) / 0.14, 0, 1));
            cx = hoverX - 4 * pU;
            cy = hoverY - 4 * pU;
            cScale = 1 - 0.12 * pU;
          } else if (t < 0.85) {
            const u = smootherstep((t - 0.22) / (0.85 - 0.22));
            cx = lerp(hoverX, parkedX, u);
            cy = lerp(hoverY, parkedY, u);
          } else if (t < 8.95) {
            cx = parkedX;
            cy = parkedY;
          } else if (t < 9.45) {
            const u = smootherstep((t - 8.95) / (9.45 - 8.95));
            cx = lerp(parkedX, hoverX, u);
            cy = lerp(parkedY, hoverY, u);
          } else {
            cx = hoverX;
            cy = hoverY;
          }

          ctx.save();
          ctx.translate(cx, cy);
          ctx.scale(cScale, cScale);

          ctx.fillStyle = '#09090B';
          ctx.strokeStyle = COLORS.white;
          ctx.lineWidth = 3.2;
          ctx.lineJoin = 'round';
          ctx.beginPath();
          ctx.moveTo(0, 0);
          ctx.lineTo(0, 42);
          ctx.lineTo(11, 32);
          ctx.lineTo(19, 49);
          ctx.lineTo(27, 45);
          ctx.lineTo(19, 28);
          ctx.lineTo(33, 28);
          ctx.closePath();
          ctx.fill();
          ctx.stroke();

          ctx.restore();
        })();

        ctx.restore();
        updateHudDisplay(t);
      }

      // Procedural Web Audio API for Interactive Mode
      let audioCtx = null;
      let audioEnabled = true;

      function ensureAudio() {
        if (!audioCtx) {
          const AC = window.AudioContext || window.webkitAudioContext;
          if (AC) audioCtx = new AC();
        }
        if (audioCtx && audioCtx.state === 'suspended') {
          audioCtx.resume();
        }
      }

      function playTone(freq0, freq1, dur, type = 'sine', gainPeak = 0.14) {
        if (!audioEnabled || !audioCtx || audioCtx.state !== 'running') return;
        const now = audioCtx.currentTime;
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = type;
        osc.frequency.setValueAtTime(freq0, now);
        if (freq1 !== freq0) {
          osc.frequency.exponentialRampToValueAtTime(Math.max(20, freq1), now + dur);
        }
        gain.gain.setValueAtTime(0.0001, now);
        gain.gain.linearRampToValueAtTime(gainPeak, now + 0.005);
        gain.gain.exponentialRampToValueAtTime(0.0001, now + dur);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start(now);
        osc.stop(now + dur + 0.01);
      }

      const SFX_CUES = [
        { t: 0.10, fn: () => playTone(980, 440, 0.055, 'triangle', 0.18) },
        { t: 0.50, fn: () => playTone(220, 95, 0.09, 'sine', 0.20) },
        { t: 2.10, fn: () => playTone(880, 1174, 0.09, 'triangle', 0.15) },
        { t: 2.50, fn: () => playTone(320, 140, 0.09, 'sine', 0.18) },
        { t: 3.00, fn: () => playTone(280, 120, 0.09, 'sine', 0.18) },
        { t: 3.50, fn: () => playTone(250, 110, 0.09, 'sine', 0.18) },
        { t: 4.00, fn: () => playTone(660, 330, 0.05, 'triangle', 0.15) },
        { t: 4.50, fn: () => playTone(587, 293, 0.05, 'triangle', 0.15) },
        { t: 5.00, fn: () => playTone(740, 370, 0.05, 'triangle', 0.15) },
        { t: 5.50, fn: () => playTone(660, 330, 0.05, 'triangle', 0.15) },
        { t: 6.00, fn: () => playTone(880, 440, 0.06, 'triangle', 0.16) },
        { t: 7.00, fn: () => playTone(260, 110, 0.12, 'triangle', 0.22) },
        { t: 7.62, fn: () => playTone(440, 880, 0.14, 'sine', 0.16) },
        { t: 8.80, fn: () => playTone(740, 370, 0.08, 'triangle', 0.18) }
      ];

      function triggerRealtimeAudio(prevT, currT) {
        if (!audioEnabled || !audioCtx || audioCtx.state !== 'running') return;
        if (currT < prevT) prevT = 0;
        SFX_CUES.forEach((cue) => {
          if (prevT < cue.t && currT >= cue.t) {
            cue.fn();
          }
        });
      }

      const btnPlay = document.getElementById('btn-play');
      const btnPrev = document.getElementById('btn-prev');
      const btnNext = document.getElementById('btn-next');
      const btnAudio = document.getElementById('btn-audio');
      const scrubber = document.getElementById('scrubber');
      const lblTime = document.getElementById('lbl-time');
      const lblFrame = document.getElementById('lbl-frame');
      const lblBeat = document.getElementById('lbl-beat');
      const pillBtns = document.querySelectorAll('.pill-btn');

      let isPlaying = params.get('clean') !== '1';
      let currentTime = 0.0;
      let lastPerf = performance.now();

      function updateHudDisplay(t) {
        if (!lblTime) return;
        lblTime.textContent = t.toFixed(3) + 's';
        const f = Math.round(t * FPS) % 600;
        lblFrame.textContent = String(f).padStart(3, '0');
        const b = Math.min(20, Math.floor(t / BEAT_SEC) + 1);
        lblBeat.textContent = String(b).padStart(2, '0');
        if (document.activeElement !== scrubber) {
          scrubber.value = t.toFixed(3);
        }
      }

      function tick(now) {
        if (isPlaying) {
          const dt = Math.min(0.1, (now - lastPerf) / 1000);
          const nextT = (currentTime + dt) % DURATION;
          triggerRealtimeAudio(currentTime, nextT);
          currentTime = nextT;
          seek(currentTime);
        }
        lastPerf = now;
        requestAnimationFrame(tick);
      }

      btnPlay.addEventListener('click', () => {
        ensureAudio();
        isPlaying = !isPlaying;
        btnPlay.textContent = isPlaying ? 'Pause' : 'Play';
        btnPlay.classList.toggle('active', isPlaying);
        lastPerf = performance.now();
      });

      btnPrev.addEventListener('click', () => {
        isPlaying = false;
        btnPlay.textContent = 'Play';
        btnPlay.classList.remove('active');
        currentTime = (currentTime - 1 / FPS + DURATION) % DURATION;
        seek(currentTime);
      });

      btnNext.addEventListener('click', () => {
        isPlaying = false;
        btnPlay.textContent = 'Play';
        btnPlay.classList.remove('active');
        currentTime = (currentTime + 1 / FPS) % DURATION;
        seek(currentTime);
      });

      btnAudio.addEventListener('click', () => {
        ensureAudio();
        audioEnabled = !audioEnabled;
        btnAudio.textContent = audioEnabled ? 'Audio: ON' : 'Audio: OFF';
        btnAudio.classList.toggle('active', audioEnabled);
      });

      scrubber.addEventListener('input', (e) => {
        isPlaying = false;
        btnPlay.textContent = 'Play';
        btnPlay.classList.remove('active');
        currentTime = parseFloat(e.target.value) || 0;
        seek(currentTime);
      });

      pillBtns.forEach((btn) => {
        btn.addEventListener('click', () => {
          ensureAudio();
          const target = parseFloat(btn.getAttribute('data-seek')) || 0;
          currentTime = target;
          seek(currentTime);
        });
      });

      window.addEventListener('pointerdown', () => {
        ensureAudio();
      }, { once: true });

      window.seek = function (t) {
        isPlaying = false;
        currentTime = t;
        seek(t);
      };

      seek(0.0);
      if (isPlaying) {
        requestAnimationFrame(tick);
      }
    })();
  </script>
</body>
</html>
"""

with open("artifacts/file_generation/ttl=63d/intermediate/ui_motion_film.html", "w") as f:
    f.write(HTML_CONTENT)

with open("artifacts/file_generation/ttl=63d/output/ui_motion_film.html", "w") as f:
    f.write(HTML_CONTENT)

print("Updated ui_motion_film.html with stepped ISO return key and card timing.")
