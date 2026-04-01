import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import wfdb
import tempfile
import os
import torch

from utils import (
    get_device,
    load_model,
    process_record,
    process_uploaded_signal,
    predict_beats,
    get_flagged,
    get_summary_counts,
    CONDITION_COLORS,
    CONDITION_LABELS,
    CONDITION_INFO,
)

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="EKG Analyzer",
    page_icon="🫀",
    layout="wide"
)

# ── Load model once at startup ─────────────────────────────────
@st.cache_resource
def get_model():
    device = get_device()
    model, class_names = load_model('./models/ecg_cnn.pth', device)
    return model, class_names, device

model, class_names, device = get_model()


# ── Downsample helper ──────────────────────────────────────────
def downsample_signal(signal, fs, target_hz=100):
    factor      = max(1, int(fs / target_hz))
    downsampled = signal[::factor]
    new_fs      = fs / factor
    return downsampled, new_fs


# ── Plotly chart builder ───────────────────────────────────────
def build_plotly_chart(signal, results, fs,
                       start_sec, duration_sec,
                       confidence_threshold, title):

    total_samples = len(signal)
    time          = np.arange(total_samples) / fs

    fig = go.Figure()

    # ── Main ECG trace ─────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=time,
        y=signal,
        mode='lines',
        line=dict(color='#2c3e50', width=0.8),
        name='ECG signal',
        hoverinfo='skip'
    ))

    # ── Build highlight bands efficiently ──────────────────────
    # Group flagged beats by class then draw one filled trace per class
    # This replaces hundreds of individual vrect calls with one path per class

    from collections import defaultdict
    class_segments = defaultdict(list)

    for r in results:
        if r['confidence'] < confidence_threshold:
            continue
        if r['prediction'] == 'Normal':
            continue
        class_segments[r['prediction']].append(r)

    y_min = float(np.min(signal))
    y_max = float(np.max(signal))

    for condition, beats in class_segments.items():
        color = CONDITION_COLORS.get(condition, '#95a5a6')

        # Build x and y arrays for a filled rectangle trace
        # Each beat contributes 5 points: left-bottom, left-top,
        # right-top, right-bottom, None (breaks the path between beats)
        x_fill = []
        y_fill = []

        for r in beats:
            x0 = r['sample_start'] / fs
            x1 = r['sample_end']   / fs
            x_fill += [x0, x0, x1, x1, None]
            y_fill += [y_min, y_max, y_max, y_min, None]

        fig.add_trace(go.Scatter(
            x=x_fill,
            y=y_fill,
            fill='toself',
            fillcolor=color,
            opacity=0.2,
            mode='none',            # no line, just fill
            name=CONDITION_LABELS.get(condition, condition),
            legendgroup=condition,
            showlegend=True,
            hoverinfo='skip',
        ))

    # ── R-peak markers with hover tooltips ─────────────────────
    # Group all markers by class for one trace per class
    for condition, beats in class_segments.items():
        color     = CONDITION_COLORS.get(condition, '#95a5a6')
        full_name = CONDITION_LABELS.get(condition, condition)

        x_markers = []
        y_markers = []
        hover_text = []

        for r in beats:
            peak_idx = min(r['r_peak'], len(signal) - 1)
            x_markers.append(r['r_peak'] / fs)
            y_markers.append(float(signal[peak_idx]))
            hover_text.append(
                f"<b>{full_name}</b><br>"
                f"Time: {r['r_peak']/fs:.2f}s<br>"
                f"Confidence: {r['confidence']:.0%}"
            )

        fig.add_trace(go.Scatter(
            x=x_markers,
            y=y_markers,
            mode='markers',
            marker=dict(color=color, size=7, symbol='triangle-down'),
            name=full_name,
            legendgroup=condition,
            showlegend=False,       # legend already added by fill trace
            hovertemplate="%{text}<extra></extra>",
            text=hover_text,
        ))

    # ── Layout ─────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude (mV)",
        hovermode='closest',
        height=380,
        margin=dict(l=60, r=20, t=50, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            range=[start_sec, start_sec + duration_sec],
            rangeslider=dict(visible=True),
            gridcolor='#ecf0f1'
        ),
        yaxis=dict(
            gridcolor='#ecf0f1',
            fixedrange=False
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0
        )
    )

    return fig

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.title("🫀 EKG Analyzer")
    st.caption("CNN-based arrhythmia detection")
    st.divider()

    input_mode = st.radio(
        "Input source",
        ["MIT-BIH record number", "Upload .dat file", "Upload ECG image (JPG/PNG)"],
        index=0
    )

    st.divider()
    st.subheader("Display settings")

    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.5,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Only flag beats above this confidence level"
    )

    duration_sec = st.slider(
        "Window duration (seconds)",
        min_value=10,
        max_value=60,
        value=30,
        step=5
    )

    st.divider()
    st.caption(f"Model running on: `{device}`")
    st.caption(f"Classes: {', '.join(class_names)}")


# ── Main panel ─────────────────────────────────────────────────
st.title("EKG Arrhythmia Analysis")
st.caption(
    "This tool flags beats for clinical review. "
    "It is not a substitute for physician interpretation."
)

# ── Initialize session state ───────────────────────────────────
if 'signal' not in st.session_state:
    st.session_state.signal   = None
if 'fs' not in st.session_state:
    st.session_state.fs       = None
if 'r_peaks' not in st.session_state:
    st.session_state.r_peaks  = None
if 'results' not in st.session_state:
    st.session_state.results  = None
if 'record_label' not in st.session_state:
    st.session_state.record_label = None


# ── Input handling ─────────────────────────────────────────────
if input_mode == "MIT-BIH record number":
    col1, col2 = st.columns([2, 1])
    with col1:
        record_id = st.text_input(
            "Record number", value="119",
            placeholder="e.g. 100, 119, 200, 208"
        )
    with col2:
        st.write("")
        st.write("")
        run_btn = st.button("Analyze", type="primary", width='stretch')

    if run_btn:
        record_path = f'./data/mitdb/{record_id}'
        if not os.path.exists(record_path + '.hea'):
            st.error(f"Record {record_id} not found in ./data/mitdb/. "
                     "Check the record number and try again.")
        else:
            with st.spinner("Loading and analyzing record..."):
                signal, fs, r_peaks = process_record(record_path)
                st.session_state.signal       = signal
                st.session_state.fs           = fs
                st.session_state.r_peaks      = r_peaks
                st.session_state.results      = None   # reset so inference reruns
                st.session_state.record_label = f"Record {record_id}"

elif input_mode == "Upload .dat file":
    st.info(
        "Upload both the `.dat` and `.hea` files for your record. "
        "These come as a pair from PhysioNet."
    )
    uploaded_files = st.file_uploader(
        "Upload ECG files",
        type=['dat', 'hea'],
        accept_multiple_files=True
    )

    if uploaded_files and len(uploaded_files) == 2:
        with st.spinner("Processing uploaded files..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                for f in uploaded_files:
                    with open(os.path.join(tmpdir, f.name), 'wb') as out:
                        out.write(f.read())

                base_name = [f.name for f in uploaded_files
                             if f.name.endswith('.hea')][0].replace('.hea', '')
                record_path = os.path.join(tmpdir, base_name)

                signal, fs, r_peaks = process_record(record_path)
                st.session_state.signal       = signal
                st.session_state.fs           = fs
                st.session_state.r_peaks      = r_peaks
                st.session_state.results      = None
                st.session_state.record_label = base_name

    elif uploaded_files and len(uploaded_files) != 2:
        st.warning("Please upload both the .dat and .hea files together.")

elif input_mode == "Upload ECG image (JPG/PNG)":
    st.info(
        "Upload a JPG or PNG of an ECG printout. "
        "For best results use a flat, well-lit scan rather than a photo."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_image = st.file_uploader(
            "Upload ECG image",
            type=['jpg', 'jpeg', 'png']
        )
    with col2:
        recording_duration = st.number_input(
            "Recording duration (seconds)",
            min_value=1,
            max_value=60,
            value=10,
            help="How many seconds of ECG are shown in the image"
        )

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded ECG", width=True)

        with st.expander("Show image processing steps"):
            import cv2
            from PIL import Image
            from image_processing import preprocess_image, remove_grid

            pil_image   = Image.open(uploaded_image).convert('RGB')
            img_array   = np.array(pil_image)
            img_bgr     = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            gray        = preprocess_image(img_bgr)
            signal_only = remove_grid(gray)

            c1, c2, c3 = st.columns(3)
            c1.image(img_array,   caption="Original",     width=True)
            c2.image(gray,        caption="Preprocessed", width=True)
            c3.image(signal_only, caption="Grid removed",  width=True)

        run_img_btn = st.button("Analyze image", type="primary")

        if run_img_btn:
            with st.spinner("Extracting signal from image..."):
                import cv2
                from PIL import Image
                from utils import process_image_upload

                pil_image = Image.open(uploaded_image).convert('RGB')
                img_array = np.array(pil_image)
                img_bgr   = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                try:
                    signal, fs, r_peaks = process_image_upload(
                        img_bgr,
                        recording_duration_sec=recording_duration
                    )
                    st.session_state.signal       = signal
                    st.session_state.fs           = fs
                    st.session_state.r_peaks      = r_peaks
                    st.session_state.results      = None
                    st.session_state.record_label = uploaded_image.name
                    st.success(
                        f"Signal extracted — {len(r_peaks)} beats detected "
                        f"over {recording_duration}s"
                    )
                except ValueError as e:
                    st.error(f"Could not extract signal: {e}")
                    st.stop()


# ── Analysis and display ───────────────────────────────────────
if st.session_state.signal is not None:

    # Run inference only if results don't exist yet for this signal
    if st.session_state.results is None:
        progress_bar = st.progress(0, text="Running inference...")
        status_text = st.empty()

        results = []
        total_peaks = len(st.session_state.r_peaks)

        pre = int(0.2 * st.session_state.fs)
        post = int(0.4 * st.session_state.fs)

        for i, peak in enumerate(st.session_state.r_peaks):
            if peak - pre < 0 or peak + post >= len(st.session_state.signal):
                continue

            beat = st.session_state.signal[peak - pre: peak + post]
            mean = beat.mean()
            std = beat.std() + 1e-8
            beat = (beat - mean) / std

            tensor = torch.FloatTensor(beat).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(tensor)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                pred_idx = probs.argmax()
                confidence = float(probs[pred_idx])
                prediction = class_names[pred_idx]

            results.append({
                'sample_start': int(peak - pre),
                'sample_end': int(peak + post),
                'r_peak': int(peak),
                'prediction': prediction,
                'confidence': confidence,
                'all_probs': {c: float(p) for c, p in zip(class_names, probs)}
            })

            # Update progress every 50 beats
            if i % 50 == 0:
                pct = int(i / total_peaks * 100)
                progress_bar.progress(pct, text=f"Analyzing beats... {i}/{total_peaks}")

        progress_bar.progress(100, text="Inference complete")
        status_text.empty()
        st.session_state.results = results

    # Pull everything out of session state for the rest of the display code
    signal   = st.session_state.signal
    fs       = st.session_state.fs
    results  = st.session_state.results

    flagged = get_flagged(results, confidence_threshold)
    counts  = get_summary_counts(results, confidence_threshold)
    total   = len(results)

    # ── Metric cards ───────────────────────────────────────────
    st.divider()
    cols = st.columns(5)
    metrics = [
        ("Total beats",  total,               None),
        ("Flagged",      len(flagged),         f"{len(flagged)/total*100:.1f}%"),
        ("PVCs",         counts.get('PVC', 0), None),
        ("PACs",         counts.get('PAC', 0), None),
        ("Other",        sum(v for k, v in counts.items()
                             if k not in ('PVC', 'PAC')), None),
    ]
    for col, (label, value, delta) in zip(cols, metrics):
        col.metric(label, value, delta)

    # ── Downsample for display then build chart ─────────────────
    st.divider()

    display_signal, display_fs = downsample_signal(signal, fs, target_hz=100)

    scale = display_fs / fs
    display_results = []
    for r in results:
        display_results.append({
            **r,
            'sample_start': int(r['sample_start'] * scale),
            'sample_end':   int(r['sample_end']   * scale),
            'r_peak':       int(r['r_peak']        * scale),
        })

    fig = build_plotly_chart(
        display_signal, display_results, display_fs,
        start_sec=0,
        duration_sec=duration_sec,
        confidence_threshold=confidence_threshold,
        title=f"EKG Analysis — {st.session_state.record_label}"
    )
    st.plotly_chart(fig, width='stretch')

    # ── Condition info cards ───────────────────────────────────
    if counts:
        st.divider()
        st.subheader("Detected conditions")

        for condition, count in counts.most_common():
            info = CONDITION_INFO.get(condition, "")
            with st.expander(
                f"**{CONDITION_LABELS.get(condition, condition)}** "
                f"— {count} beat{'s' if count != 1 else ''} flagged"
            ):
                if info:
                    st.write(info)
                st.caption("Sample predictions:")
                sample_beats = [r for r in flagged
                                if r['prediction'] == condition][:3]
                for r in sample_beats:
                    t = r['r_peak'] / fs
                    st.write(f"**{t:.2f}s** — confidence {r['confidence']:.0%}")
                    prob_df = pd.DataFrame({
                        'Class':       list(r['all_probs'].keys()),
                        'Probability': [f"{v:.1%}"
                                        for v in r['all_probs'].values()]
                    })
                    st.dataframe(prob_df, hide_index=True, width='stretch')

    # ── Flagged beats table ────────────────────────────────────
    st.divider()
    st.subheader("Flagged beats — full list")

    if flagged:
        rows = []
        for r in flagged:
            t = r['r_peak'] / fs
            rows.append({
                'Time (s)':   f"{t:.2f}",
                'Time (m:s)': f"{int(t//60)}:{int(t%60):02d}",
                'Condition':  CONDITION_LABELS.get(r['prediction'],
                                                   r['prediction']),
                'Confidence': f"{r['confidence']:.0%}",
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, width='stretch', hide_index=True)

        csv = df.to_csv(index=False)
        st.download_button(
            label="Download flagged beats as CSV",
            data=csv,
            file_name="flagged_beats.csv",
            mime="text/csv",
            key="download_csv_btn"
        )
    else:
        st.success(
            "No abnormal beats flagged above the confidence threshold. "
            "Try lowering the threshold in the sidebar if you expect findings."
        )

    # ── Metric cards ───────────────────────────────────────────
    st.divider()
    cols = st.columns(5)
    metrics = [
        ("Total beats",  total,               None),
        ("Flagged",      len(flagged),         f"{len(flagged)/total*100:.1f}%"),
        ("PVCs",         counts.get('PVC', 0), None),
        ("PACs",         counts.get('PAC', 0), None),
        ("Other",        sum(v for k, v in counts.items()
                             if k not in ('PVC', 'PAC')), None),
    ]
    for col, (label, value, delta) in zip(cols, metrics):
        col.metric(label, value, delta)

    with st.expander("Debug info"):
        st.write("Signal shape:", st.session_state.signal.shape
        if st.session_state.signal is not None else None)
        st.write("R-peaks:", len(st.session_state.r_peaks)
        if st.session_state.r_peaks is not None else None)
        st.write("Results count:", len(st.session_state.results)
        if st.session_state.results is not None else None)
        st.write("Record label:", st.session_state.record_label)

    # ── Downsample for display then build chart ─────────────────
    st.divider()

    display_signal, display_fs = downsample_signal(signal, fs, target_hz=100)

    scale = display_fs / fs
    display_results = []
    for r in results:
        display_results.append({
            **r,
            'sample_start': int(r['sample_start'] * scale),
            'sample_end':   int(r['sample_end']   * scale),
            'r_peak':       int(r['r_peak']        * scale),
        })

    fig = build_plotly_chart(
        display_signal, display_results, display_fs,
        start_sec=0,
        duration_sec=duration_sec,
        confidence_threshold=confidence_threshold,
        title="EKG Analysis"
    )
    st.plotly_chart(fig, width='stretch')

    # ── Condition info cards ───────────────────────────────────
    if counts:
        st.divider()
        st.subheader("Detected conditions")

        for condition, count in counts.most_common():
            color = CONDITION_COLORS.get(condition, '#95a5a6')
            info  = CONDITION_INFO.get(condition, "")
            with st.expander(
                f"**{CONDITION_LABELS.get(condition, condition)}** "
                f"— {count} beat{'s' if count != 1 else ''} flagged"
            ):
                if info:
                    st.write(info)
                st.caption("Sample predictions:")
                sample_beats = [r for r in flagged
                                if r['prediction'] == condition][:3]
                for r in sample_beats:
                    t = r['r_peak'] / fs
                    st.write(f"**{t:.2f}s** — confidence {r['confidence']:.0%}")
                    prob_df = pd.DataFrame({
                        'Class':       list(r['all_probs'].keys()),
                        'Probability': [f"{v:.1%}"
                                        for v in r['all_probs'].values()]
                    })
                    st.dataframe(prob_df, hide_index=True, width='stretch')

    # ── Flagged beats table ────────────────────────────────────
    st.divider()
    st.subheader("Flagged beats — full list")

    if flagged:
        rows = []
        for r in flagged:
            t = r['r_peak'] / fs
            rows.append({
                'Time (s)':   f"{t:.2f}",
                'Time (m:s)': f"{int(t//60)}:{int(t%60):02d}",
                'Condition':  CONDITION_LABELS.get(r['prediction'],
                                                   r['prediction']),
                'Confidence': f"{r['confidence']:.0%}",
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, width='stretch', hide_index=True)

        csv = df.to_csv(index=False)
        st.download_button(
            label="Download flagged beats as CSV",
            data=csv,
            file_name="flagged_beats.csv",
            mime="text/csv"
        )
    else:
        st.success(
            "No abnormal beats flagged above the confidence threshold. "
            "Try lowering the threshold in the sidebar if you expect findings."
        )