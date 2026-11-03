using System.Globalization;
using UnityEngine;
using UnityEngine.UI;

namespace SPWM.Visualization
{
    /// <summary>
    /// Streamlined telemetry HUD bound to <see cref="GhostPlaybackManager"/>.
    /// Displays only the Title, Current Frame/Phase, and three moving error graphs.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class TelemetryDisplay : MonoBehaviour
    {
        [Header("Source")]
        [SerializeField] private GhostPlaybackManager playback;

        [Header("Title & Phase")]
        [SerializeField] private Text titleText;
        [SerializeField] private Text frameAndPhaseText;

        [Header("Error Graphs")]
        [SerializeField] private UISparkline latentErrorSparkline;
        [SerializeField] private UISparkline positionErrorSparkline;
        [SerializeField] private UISparkline velocityErrorSparkline;

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
            if (playback != null && playback.IsLoaded) Bind(playback);
        }

        public void Bind(GhostPlaybackManager pm)
        {
            // 2.1 Title
            if (titleText != null) 
            {
                titleText.text = "Spiking Predictor World Model";
            }

            GhostTrajectoryData data = pm.Data;
            
            if (data != null && data.FrameCount > 0)
            {
                var latentSeries = new float[data.FrameCount];
                var posSeries = new float[data.FrameCount];
                var velSeries = new float[data.FrameCount];

                // Pre-fill series for the moving graphs
                for (int i = 0; i < data.FrameCount; i++)
                {
                    var metrics = data.frames[i].instantaneous_metrics;
                    
                    // Note: Assuming 'latent_error' has been added to GhostInstantaneousMetrics
                    latentSeries[i] = metrics.latent_error; 
                    posSeries[i] = metrics.position_abs_error;
                    velSeries[i] = metrics.velocity_abs_error; 
                }

                if (latentErrorSparkline != null) latentErrorSparkline.SetSeries(latentSeries);
                if (positionErrorSparkline != null) positionErrorSparkline.SetSeries(posSeries);
                if (velocityErrorSparkline != null) velocityErrorSparkline.SetSeries(velSeries);
            }
            
            OnRestart(pm);
        }

        private void OnRestart(GhostPlaybackManager pm)
        {
            _lastStep = -1;
            Refresh(force: true);
        }

        private void LateUpdate() => Refresh(force: false);

        private void Refresh(bool force)
        {
            if (playback == null || !playback.IsLoaded) return;

            float framePos = playback.FramePosition;
            int currentFrame = playback.CurrentStep;

            // 2.2 Current frame - current phase
            if (currentFrame != _lastStep || force)
            {
                if (frameAndPhaseText != null)
                {
                    frameAndPhaseText.text = $"{currentFrame} - {GetPhaseName(framePos)}";
                }
                _lastStep = currentFrame;
            }

            // 2.3, 2.4, 2.5 Map errors onto moving graphs continuously
            if (latentErrorSparkline != null) latentErrorSparkline.SetPlayhead(framePos);
            if (positionErrorSparkline != null) positionErrorSparkline.SetPlayhead(framePos);
            if (velocityErrorSparkline != null) velocityErrorSparkline.SetPlayhead(framePos);
        }

        private string GetPhaseName(float framePos)
        {
            if (framePos < 4f) return "burn in";
            
            int heftStart = playback.HeftStartFrame;
            int graspLock = playback.GraspLockFrame;
            int sensoryRelease = playback.SensoryReleaseFrame;

            // Ground truth sensory injection phase spanning pre-contact approach, mid-contact grasp lock, and post-contact lift
            if (heftStart >= 0 && sensoryRelease >= 0 && framePos >= heftStart && framePos <= sensoryRelease)
            {
                if (graspLock >= 0)
                {
                    if (framePos < graspLock) return "ground truth pre";
                    if (framePos == graspLock) return "ground truth mid";
                    return "ground truth post";
                }
                return "ground truth partial";
            }
            
            return "autonomous";
        }
    }
}
