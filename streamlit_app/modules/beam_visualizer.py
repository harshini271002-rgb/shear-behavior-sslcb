import streamlit as st
import streamlit.components.v1 as components

def show():
    st.header("👁️ Beam Section Visualizer")
    st.markdown("Interactive real-time visualization of beam behavior, stress distribution, and failure modes.")

    # --- Sidebar Inputs ---
    st.sidebar.markdown("### 🔧 Section Dimensions")
    
    defaults = st.session_state.get('predictor_inputs', {})
    def_D = float(defaults.get('D', 200.0))
    def_d1 = float(defaults.get('d1', def_D - 40.0 if 'D' in defaults else 160.0)) 
    def_tw = float(defaults.get('tw', 2.0))
    def_b = float(defaults.get('b', 70.0))
    def_ratio = float(defaults.get('dwh/d1', 0.5))

    D = st.sidebar.slider("Total Depth (D) [mm]", 100.0, 400.0, def_D, 10.0)
    d1 = st.sidebar.slider("Web Depth (d1) [mm]", 50.0, 350.0, def_d1, 5.0) 
    tw = st.sidebar.slider("Web Thickness (tw) [mm]", 1.0, 10.0, def_tw, 0.5)
    b = st.sidebar.slider("Flange Width (b) [mm]", 30.0, 150.0, def_b, 5.0)
    
    st.sidebar.markdown("### 🕳️ Opening Parameters")
    opening_ratio = st.sidebar.slider("Opening Ratio (dwh/d1)", 0.0, 0.85, def_ratio, 0.05)
    
    lip = 15.0 
    real_capacity = st.session_state.get('last_prediction', 100 * (1.0 - (opening_ratio*0.6)))
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin: 0; font-family: 'Segoe UI', Arial, sans-serif; background-color: transparent; overflow-x: hidden; }}
            .main-grid {{ display: flex; flex-direction: column; gap: 40px; padding: 20px; }}
            .viz-card {{
                background: white; border: 2.5px solid #dfe6e9; border-radius: 16px;
                padding: 30px; box-shadow: 0 10px 25px rgba(0,0,0,0.1);
                display: flex; flex-direction: column; align-items: center;
                width: 90%; max-width: 1000px; margin: 0 auto;
            }}
            h3 {{
                margin: 0 0 25px 0; color: #2d3436; font-size: 24px; font-weight: 800;
                text-transform: uppercase; border-bottom: 5px solid #0984e3;
                padding-bottom: 10px; width: 100%; text-align: center;
            }}
            canvas {{ background: #fff; width: 100%; height: 450px; border-radius: 10px; display: block; }}
            .legend {{ display: flex; gap: 30px; font-size: 18px; margin-top: 25px; font-weight: bold; color: #2d3436; text-align: center; flex-wrap: wrap; justify-content: center; }}
            .fea-gradient {{
                width: 250px; height: 15px; 
                background: linear-gradient(to right, #0984e3, #00cec9, #ffeaa7, #fab1a0, #d63031);
                border-radius: 5px; margin: 0 15px;
            }}
        </style>
    </head>
    <body>
    <div class="main-grid">
        <div class="viz-card">
            <h3>Geometric Section (C-Channel) [CRT]</h3>
            <canvas id="canvasGeo"></canvas>
            <div class="legend">D:{D}mm | b:{b}mm | tw:{tw}mm | r:{(d1*opening_ratio/2):.1f}mm</div>
        </div>
        <div class="viz-card">
            <h3>FEA Shear Stress Distribution [τ]</h3>
            <canvas id="canvasStress"></canvas>
            <div class="legend" style="align-items:center;">Min <div class="fea-gradient"></div> Max</div>
        </div>
        <div class="viz-card">
            <h3>Failure Mechanism (Vierendeel)</h3>
            <canvas id="canvasFail"></canvas>
            <div class="legend">Orange: Plastic Hinges | Path: Shear Buckling</div>
        </div>
        <div class="viz-card">
            <h3>Capacity Response Curve</h3>
            <canvas id="canvasLoad"></canvas>
            <div class="legend" style="color:#0984e3;">Ultimate Capacity: {real_capacity:.2f} kN</div>
        </div>
    </div>

    <script>
        const SCALE = 2; 
        const D = {D}, d1 = {d1}, tw = {tw}, b = {b}, ratio = {opening_ratio}, lip = {lip}, cap = {real_capacity};

        function setup(id) {{
            const c = document.getElementById(id);
            const ctx = c.getContext('2d');
            
            // Safer dimension reading
            let w = c.clientWidth || 800;
            let h = c.clientHeight || 450;
            
            c.width = w * SCALE;
            c.height = h * SCALE;
            ctx.setTransform(1, 0, 0, 1, 0, 0);
            ctx.scale(SCALE, SCALE);
            return {{ ctx, w, h }};
        }}

        function drawGeo() {{
            const {{ ctx, w, h }} = setup('canvasGeo');
            const pad = 60;
            const s = Math.min((w - 2.5*pad)/b, (h - 2*pad)/D);
            const x = (w - b*s)/2, y = (h - D*s)/2;
            const tw_s = tw*s, tf_s = tw*1.5*s, lip_s = lip*s;

            ctx.clearRect(0,0,w,h);
            ctx.beginPath();
            ctx.moveTo(x + b*s, y + lip_s); ctx.lineTo(x + b*s, y); ctx.lineTo(x, y); ctx.lineTo(x, y + D*s);
            ctx.lineTo(x + b*s, y + D*s); ctx.lineTo(x + b*s, y + D*s - lip_s);
            ctx.lineTo(x + b*s - tw_s, y + D*s - lip_s); ctx.lineTo(x + b*s - tw_s, y + D*s - tf_s);
            ctx.lineTo(x + tw_s, y + D*s - tf_s); ctx.lineTo(x + tw_s, y + tf_s);
            ctx.lineTo(x + b*s - tw_s, y + tf_s); ctx.lineTo(x + b*s - tw_s, y + lip_s);
            ctx.closePath();
            ctx.fillStyle = '#f1f2f6'; ctx.fill(); 
            ctx.strokeStyle = '#0984e3'; ctx.lineWidth = 4; ctx.stroke();

            if(ratio > 0.01) {{
                const r_s = (d1 * ratio * s) / 2;
                ctx.setLineDash([8, 4]); ctx.strokeStyle = '#d63031'; ctx.lineWidth = 2;
                ctx.strokeRect(x, y + (D*s)/2 - r_s, tw_s, r_s*2); ctx.setLineDash([]);
            }}
            ctx.fillStyle='#2d3436'; ctx.font='bold 18px Arial'; ctx.textAlign='center';
            ctx.fillText(D+"mm", x + b*s + 50, y + D*s/2);
            ctx.fillText(b+"mm", x + b*s/2, y - 25);
        }}

        function getHoles(w, h, r) {{
            const bL = w * 0.8;
            const spacing = Math.max(bL * 0.45, r * 2.5);
            return {{ x1: w/2 - spacing/2, x2: w/2 + spacing/2, r }};
        }}

        function drawStress() {{
            const {{ ctx, w, h }} = setup('canvasStress');
            const bH = 120, y0 = (h - bH)/2, bL = w * 0.85, x0 = (w-bL)/2;
            const hI = getHoles(w, h, (bH * 0.8 * ratio)/2);
            
            ctx.clearRect(0,0,w,h);
            const g = ctx.createLinearGradient(x0, 0, x0+bL, 0);
            g.addColorStop(0, '#0984e3'); g.addColorStop(0.5, '#fab1a0'); g.addColorStop(1, '#0984e3');
            ctx.fillStyle=g; ctx.globalAlpha=0.15; ctx.fillRect(x0, y0, bL, bH); ctx.globalAlpha=1;

            if(ratio > 0.05) {{
                [hI.x1, hI.x2].forEach(x => {{
                    const bg = ctx.createLinearGradient(x-hI.r*2, y0+bH, x+hI.r*2, y0);
                    bg.addColorStop(0, 'transparent'); bg.addColorStop(0.5, 'rgba(214, 48, 49, 0.4)'); bg.addColorStop(1, 'transparent');
                    ctx.fillStyle=bg; ctx.beginPath(); ctx.moveTo(x-hI.r*3.5, y0+bH); ctx.lineTo(x-hI.r, y0+bH); ctx.lineTo(x+hI.r*3.5, y0); ctx.lineTo(x+hI.r, y0); ctx.fill();
                }});
                ctx.globalCompositeOperation = 'destination-out';
                ctx.beginPath(); 
                ctx.arc(hI.x1, y0+bH/2, hI.r, 0, Math.PI*2); 
                ctx.arc(hI.x2, y0+bH/2, hI.r, 0, Math.PI*2); ctx.fill();
                ctx.globalCompositeOperation = 'source-over';
                
                ctx.strokeStyle='#2d3436'; ctx.lineWidth=2;
                ctx.beginPath(); ctx.arc(hI.x1, y0+bH/2, hI.r, 0, Math.PI*2); ctx.stroke();
                ctx.beginPath(); ctx.arc(hI.x2, y0+bH/2, hI.r, 0, Math.PI*2); ctx.stroke();
            }}
            ctx.strokeStyle='#0984e3'; ctx.lineWidth=5; ctx.strokeRect(x0, y0, bL, bH);
        }}

        function drawFail() {{
            const {{ ctx, w, h }} = setup('canvasFail');
            const bH = 120, y0 = (h - bH)/2, bL = w * 0.85, x0 = (w-bL)/2;
            const r = (bH * 0.8 * ratio)/2;
            const hI = getHoles(w, h, r);
            const d = 35 * ratio;

            ctx.clearRect(0,0,w,h);
            if(ratio > 0.05) {{
                ctx.beginPath(); ctx.moveTo(x0, y0);
                ctx.lineTo(hI.x1-r, y0); ctx.bezierCurveTo(hI.x1, y0+d, hI.x1, y0+d, hI.x1+r, y0);
                ctx.lineTo(hI.x2-r, y0); ctx.bezierCurveTo(hI.x2, y0+d, hI.x2, y0+d, hI.x2+r, y0);
                ctx.lineTo(x0+bL, y0); ctx.lineTo(x0+bL, y0+bH);
                ctx.lineTo(hI.x2+r, y0+bH); ctx.bezierCurveTo(hI.x2, y0+bH-d, hI.x2, y0+bH-d, hI.x2-r, y0+bH);
                ctx.lineTo(hI.x1+r, y0+bH); ctx.bezierCurveTo(hI.x1, y0+bH-d, hI.x1, y0+bH-d, hI.x1-r, y0+bH);
                ctx.lineTo(x0, y0+bH); ctx.closePath();
                ctx.fillStyle='#f1f2f6'; ctx.fill(); ctx.strokeStyle='#0984e3'; ctx.lineWidth=5; ctx.stroke();
                
                ctx.globalCompositeOperation='destination-out';
                ctx.beginPath(); ctx.arc(hI.x1, y0+bH/2+d*0.5, r, 0, Math.PI*2); ctx.arc(hI.x2, y0+bH/2+d*0.5, r, 0, Math.PI*2); ctx.fill();
                ctx.globalCompositeOperation='source-over';
                
                [hI.x1, hI.x2].forEach(x => {{
                    [[x-r, y0+4], [x+r, y0+4+d*0.5], [x-r, y0+bH-4], [x+r, y0+bH-4-d*0.5]].forEach(p => {{
                        ctx.fillStyle='#e17055'; ctx.beginPath(); ctx.arc(p[0], p[1], 12, 0, Math.PI*2); ctx.fill();
                        ctx.strokeStyle='#d63031'; ctx.lineWidth=1.5; ctx.stroke();
                    }});
                }});
            }} else {{
                ctx.strokeStyle='#0984e3'; ctx.lineWidth=5; ctx.strokeRect(x0, y0, bL, bH);
            }}
        }}

        function drawLoad() {{
            const {{ ctx, w, h }} = setup('canvasLoad');
            const p = 60, gW = w - 2*p, gH = h - 2*p;
            
            ctx.clearRect(0,0,w,h);
            ctx.strokeStyle='#ecf0f1'; ctx.lineWidth=1;
            for(let i=0; i<=10; i++) {{
                let ly = h - p - (gH/10)*i; ctx.beginPath(); ctx.moveTo(p, ly); ctx.lineTo(w-p, ly); ctx.stroke();
                let lx = p + (gW/10)*i; ctx.beginPath(); ctx.moveTo(lx, h-p); ctx.lineTo(lx, p); ctx.stroke();
            }}
            ctx.strokeStyle='#2d3436'; ctx.lineWidth=3; ctx.beginPath(); ctx.moveTo(p, p-10); ctx.lineTo(p, h-p); ctx.lineTo(w-p+10, h-p); ctx.stroke();
            
            const mx = (v) => p + (v/20)*gW, my = (v) => h-p - (v/(cap*1.3))*gH;
            ctx.beginPath(); ctx.moveTo(p, h-p); ctx.lineTo(mx(5), my(cap*0.8));
            ctx.bezierCurveTo(mx(10), my(cap*1.1), mx(14), my(cap), mx(14), my(cap));
            ctx.bezierCurveTo(mx(16), my(cap*0.8), mx(19), my(cap*0.4), mx(20), my(cap*0.2));
            ctx.strokeStyle='#0984e3'; ctx.lineWidth=8; ctx.stroke();
            ctx.fillStyle='#d63031'; ctx.beginPath(); ctx.arc(mx(14), my(cap), 12, 0, Math.PI*2); ctx.fill();
            ctx.font='bold 20px Arial'; ctx.fillText(cap.toFixed(2)+" kN", mx(14), my(cap)-25);
        }}

        function run() {{ 
            try {{ drawGeo(); drawStress(); drawFail(); drawLoad(); }} catch(e) {{ console.error(e); }}
        }}

        // Execution chain
        if(document.readyState === 'complete') {{ run(); }}
        else {{ window.addEventListener('load', run); }}
        window.addEventListener('resize', run);
        setTimeout(run, 200);
        setTimeout(run, 1000);
    </script>
    </body>
    </html>
    """
    components.html(html_code, height=2500, scrolling=True)
    st.info("📊 View updated. Scroll down to see all detailed charts.")
