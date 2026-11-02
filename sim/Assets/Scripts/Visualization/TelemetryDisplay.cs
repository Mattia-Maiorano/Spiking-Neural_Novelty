using System.Globalization;
using UnityEngine;
using UnityEngine.UI;

namespace SPWM.Visualization
{
    /// <summary>
    /// LinkedIn-ready telemetry HUD bound to <see cref="GhostPlaybackManager"/>.
    /// Displays horizon progress (t / H + bar), attractor convergence (velocity relative degradation sparkline)
    /// and phase shift (absolute joint-position error). Values are interpolated at the same continuous frame
    /// position as the rigs, so numbers and limbs move in lock-step.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class TelemetryDisplay : MonoBehaviour
    {
        [Header("Source")]
        [SerializeField] private GhostPlaybackManager playback;

        [Header("Header")]
        [SerializeField] private Text titleText;
        [SerializeField] private Text subtitleText;

        [Header("Horizon Progress")]
        [SerializeField] private Text stepCounterText;
        [SerializeField] private Text timeText;
        [Tooltip("Child RectTransform whose anchorMax.x is driven 0..1 (no sprite needed).")]
        [SerializeField] private RectTransform progressFill;

        [Header("Attractor Convergence")]
        [SerializeField] private UISparkline velocitySparkline;
        [SerializeField] private Text velocityValueText;

        [Header("Phase Shift")]
        [SerializeField] private Text positionErrorText;
        [SerializeField] private Text positionErrorDegText;
        [SerializeField] private Image positionErrorAccent;

        [Header("Formatting")]
        [SerializeField] private string titleFormat = "SPWM {0} · GHOST EXECUTION";
        [Tooltip("Position error (rad) at which the accent turns from green to amber / red.")]
        [SerializeField] private float warnPositionError = 0.5f;
        [SerializeField] private float alertPositionError = 1.0f;
        [SerializeField] private Color goodColor = new Color(0.25f, 1f, 0.75f);
        [SerializeField] private Color warnColor = new Color(1f, 0.78f, 0.25f);
        [SerializeField] private Color alertColor = new Color(1f, 0.38f, 0.42f);
        [Tooltip("Exponential smoothing of displayed numbers (seconds). 0 = raw.")]
        [SerializeField, Range(0f, 0.5f)] private float numberSmoothing = 0.08f;

        private static readonly CultureInfo Inv = CultureInfo.InvariantCulture;
        private float _shownPos, _shownVel;
        private int _lastStep = -1;

        private void OnEnable()
        {
            if (playback == null) return;
            playback.Loaded += Bind;
            playback.PlaybackRestarted += OnRestart;
            if (playback.IsLoaded) Bind(playback);
        }

        private void OnDisable()
        {
            if (playback == null) return;
            playback.Loaded -= Bind;
            playback.PlaybackRestarted -= OnRestart;
        }

        private void Start()
        {
            // Covers the case where Loaded fired in the manager's Awake before our OnEnable subscribed.
            if (playback != null && playback.IsLoaded) Bind(playback);
        }

        public void Bind(GhostPlaybackManager pm)
        {
            GhostTrajectoryData data = pm.Data;
            GhostMetadata meta = data.metadata;

            if (titleText != null) titleText.text = string.Format(Inv, titleFormat, PrettyVersion(meta.model_version));
            if (subtitleText != null)
                subtitleText.text = string.Format(Inv, "M1 autoregressive rollout · H = {0} · Δt = {1:0} ms",
                                                  meta.horizon, meta.dt * 1000f);

            if (velocitySparkline != null)
            {
                var series = new float[data.FrameCount];
                for (int i = 0; i < series.Length; i++)
                    series[i] = data.frames[i].instantaneous_metrics.velocity_relative_degradation;
                velocitySparkline.SetSeries(series);
            }
            OnRestart(pm);
        }

        private void OnRestart(GhostPlaybackManager pm)
        {
            GhostInstantaneousMetrics m = pm.SampleMetrics();
            _shownPos = m.position_abs_error;
            _shownVel = m.velocity_relative_degradation;
            _lastStep = -1;
            Refresh(force: true);
        }

        private void LateUpdate() => Refresh(force: false);

        private void Refresh(bool force)
        {
            if (playback == null || !playback.IsLoaded) return;

            GhostInstantaneousMetrics m = playback.SampleMetrics();
            float k = numberSmoothing <= 0f || force ? 1f : 1f - Mathf.Exp(-Time.deltaTime / numberSmoothing);
            _shownPos = Mathf.Lerp(_shownPos, m.position_abs_error, k);
            _shownVel = Mathf.Lerp(_shownVel, m.velocity_relative_degradation, k);

            // Horizon progress
            int step = playback.CurrentStep, horizon = playback.FrameCount;
            if (step != _lastStep || force)
            {
                if (stepCounterText != null)
                    stepCounterText.text = string.Format(Inv, "<b>{0}</b><color=#FFFFFF66> / {1}</color>", step, horizon);
                _lastStep = step;
            }
            if (timeText != null)
                timeText.text = string.Format(Inv, "t = {0:0.00} s", (playback.FramePosition + 1f) * playback.Data.Dt);
            if (progressFill != null)
            {
                Vector2 max = progressFill.anchorMax;
                max.x = playback.Progress;
                progressFill.anchorMax = max;
            }

            // Attractor convergence
            if (velocitySparkline != null) velocitySparkline.SetPlayhead(playback.FramePosition);
            if (velocityValueText != null)
                velocityValueText.text = string.Format(Inv, "×{0:0.00}", _shownVel);

            // Phase shift
            if (positionErrorText != null)
            {
                int unitSize = Mathf.Max(8, Mathf.RoundToInt(positionErrorText.fontSize * 0.5f));
                positionErrorText.text = string.Format(Inv, "{0:0.000} <size={1}>rad</size>", _shownPos, unitSize);
            }
            if (positionErrorDegText != null)
                positionErrorDegText.text = string.Format(Inv, "≈ {0:0.0}° joint-space drift", _shownPos * Mathf.Rad2Deg);
            if (positionErrorAccent != null)
                positionErrorAccent.color = SeverityColor(_shownPos);
        }

        private Color SeverityColor(float err)
        {
            if (err <= warnPositionError) return Color.Lerp(goodColor, warnColor, Mathf.InverseLerp(0f, warnPositionError, err) * 0.3f);
            if (err <= alertPositionError) return Color.Lerp(warnColor, alertColor, Mathf.InverseLerp(warnPositionError, alertPositionError, err));
            return alertColor;
        }

        private static string PrettyVersion(string v)
        {
            // "spwm_v0_4_0" -> "v0.4.0"
            if (string.IsNullOrEmpty(v)) return "";
            int idx = v.IndexOf('v');
            string tail = idx >= 0 ? v.Substring(idx) : v;
            return tail.Replace('_', '.');
        }
    }
}
