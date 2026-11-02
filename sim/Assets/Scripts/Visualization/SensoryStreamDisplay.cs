using System;
using System.Globalization;
using UnityEngine;
using UnityEngine.UI;

namespace SPWM.Visualization
{
    /// <summary>
    /// Visualizes sensory event streams and optical camera feeds (Global workspace and Egocentric Wrist)
    /// in real-time or from pre-recorded trajectory data for SPWM (schema spwm.ghost_trajectory/1.0).
    /// Depicts:
    ///   1. Global Static Camera: Optical feed + 32x32 Neuromorphic Event Stream (+1 ON / -1 OFF spikes).
    ///   2. Egocentric Wrist Camera: Wrist optical feed + 32x32 Neuromorphic Event Stream.
    ///   3. Instantaneous sensory event activity (act) and perceptual gating status (ACTIVE vs QUIESCENT).
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class SensoryStreamDisplay : MonoBehaviour
    {
        private static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

        [Header("Playback Reference")]
        [SerializeField] private GhostPlaybackManager playback;

        [Header("Cameras")]
        [SerializeField] private Camera globalCamera;
        [SerializeField] private Camera wristCamera;

        [Header("Global Sensor UI")]
        [SerializeField] private RawImage globalOpticalImage;
        [SerializeField] private RawImage globalEventsImage;
        [SerializeField] private Text globalStatusText;

        [Header("Wrist Sensor UI")]
        [SerializeField] private RawImage wristOpticalImage;
        [SerializeField] private RawImage wristEventsImage;
        [SerializeField] private Text wristStatusText;

        [Header("Sensory Header & Legend")]
        [SerializeField] private Text headerTitleText;
        [SerializeField] private Text headerSubtitleText;
        [SerializeField] private Text legendText;

        [Header("Event Detection Parameters")]
        [SerializeField] private float eventThreshold = 0.08f;
        [SerializeField] private float activityThreshold = 0.001f;
        [SerializeField] private int sensorWidth = 32;
        [SerializeField] private int sensorHeight = 32;

        [Header("Spike Colors")]
        [SerializeField] private Color32 onSpikeColor = new Color32(0, 240, 255, 255);    // Neon Cyan
        [SerializeField] private Color32 offSpikeColor = new Color32(255, 59, 92, 255);  // Neon Crimson
        [SerializeField] private Color32 backgroundColor = new Color32(11, 16, 23, 255); // Dark Slate

        // Textures & Buffers
        private RenderTexture _globalOpticalRT;
        private RenderTexture _wristOpticalRT;
        private RenderTexture _globalSensorRT;
        private RenderTexture _wristSensorRT;

        private Texture2D _globalReadbackTex;
        private Texture2D _wristReadbackTex;
        private Texture2D _globalEventTex;
        private Texture2D _wristEventTex;

        private Color32[] _globalPixels;
        private Color32[] _wristPixels;
        private float[] _prevGlobalLogL;
        private float[] _prevWristLogL;
        private bool _isFirstFrame = true;
        private int _lastFrameIndex = -1;

        public float LastGlobalActivity { get; private set; }
        public float LastWristActivity { get; private set; }
        public float LastCombinedActivity => 0.5f * (LastGlobalActivity + LastWristActivity);

        private void OnEnable()
        {
            if (playback != null)
            {
                playback.Loaded += OnPlaybackLoaded;
                playback.PlaybackRestarted += OnPlaybackRestarted;
                if (playback.IsLoaded)
                {
                    OnPlaybackLoaded(playback);
                }
            }
        }

        private void OnDisable()
        {
            if (playback != null)
            {
                playback.Loaded -= OnPlaybackLoaded;
                playback.PlaybackRestarted -= OnPlaybackRestarted;
            }
        }

        private void Start()
        {
            EnsureTextures();
            EnsureCameras();
            if (playback != null && playback.IsLoaded)
            {
                OnPlaybackLoaded(playback);
            }
        }

        private void OnDestroy()
        {
            ReleaseTextures();
        }

        private void EnsureTextures()
        {
            int numPixels = sensorWidth * sensorHeight;
            if (_globalPixels == null || _globalPixels.Length != numPixels)
            {
                _globalPixels = new Color32[numPixels];
                _wristPixels = new Color32[numPixels];
                _prevGlobalLogL = new float[numPixels];
                _prevWristLogL = new float[numPixels];
            }

            if (_globalOpticalRT == null)
            {
                _globalOpticalRT = new RenderTexture(256, 256, 16, RenderTextureFormat.ARGB32)
                {
                    name = "GlobalOpticalRT",
                    filterMode = FilterMode.Bilinear
                };
                _globalOpticalRT.Create();
            }

            if (_wristOpticalRT == null)
            {
                _wristOpticalRT = new RenderTexture(256, 256, 16, RenderTextureFormat.ARGB32)
                {
                    name = "WristOpticalRT",
                    filterMode = FilterMode.Bilinear
                };
                _wristOpticalRT.Create();
            }

            if (_globalSensorRT == null)
            {
                _globalSensorRT = new RenderTexture(sensorWidth, sensorHeight, 16, RenderTextureFormat.ARGB32)
                {
                    name = "GlobalSensorRT",
                    filterMode = FilterMode.Point
                };
                _globalSensorRT.Create();
            }

            if (_wristSensorRT == null)
            {
                _wristSensorRT = new RenderTexture(sensorWidth, sensorHeight, 16, RenderTextureFormat.ARGB32)
                {
                    name = "WristSensorRT",
                    filterMode = FilterMode.Point
                };
                _wristSensorRT.Create();
            }

            if (_globalReadbackTex == null)
            {
                _globalReadbackTex = new Texture2D(sensorWidth, sensorHeight, TextureFormat.RGBA32, false);
            }
            if (_wristReadbackTex == null)
            {
                _wristReadbackTex = new Texture2D(sensorWidth, sensorHeight, TextureFormat.RGBA32, false);
            }

            if (_globalEventTex == null)
            {
                _globalEventTex = new Texture2D(sensorWidth, sensorHeight, TextureFormat.RGBA32, false)
                {
                    name = "GlobalEventsTexture",
                    filterMode = FilterMode.Point
                };
            }
            if (_wristEventTex == null)
            {
                _wristEventTex = new Texture2D(sensorWidth, sensorHeight, TextureFormat.RGBA32, false)
                {
                    name = "WristEventsTexture",
                    filterMode = FilterMode.Point
                };
            }

            // Bind textures to UI images
            if (globalOpticalImage != null) globalOpticalImage.texture = _globalOpticalRT;
            if (globalEventsImage != null) globalEventsImage.texture = _globalEventTex;
            if (wristOpticalImage != null) wristOpticalImage.texture = _wristOpticalRT;
            if (wristEventsImage != null) wristEventsImage.texture = _wristEventTex;
        }

        private void ReleaseTextures()
        {
            if (_globalOpticalRT != null) { _globalOpticalRT.Release(); DestroyImmediate(_globalOpticalRT); _globalOpticalRT = null; }
            if (_wristOpticalRT != null) { _wristOpticalRT.Release(); DestroyImmediate(_wristOpticalRT); _wristOpticalRT = null; }
            if (_globalSensorRT != null) { _globalSensorRT.Release(); DestroyImmediate(_globalSensorRT); _globalSensorRT = null; }
            if (_wristSensorRT != null) { _wristSensorRT.Release(); DestroyImmediate(_wristSensorRT); _wristSensorRT = null; }

            if (_globalReadbackTex != null) { DestroyImmediate(_globalReadbackTex); _globalReadbackTex = null; }
            if (_wristReadbackTex != null) { DestroyImmediate(_wristReadbackTex); _wristReadbackTex = null; }
            if (_globalEventTex != null) { DestroyImmediate(_globalEventTex); _globalEventTex = null; }
            if (_wristEventTex != null) { DestroyImmediate(_wristEventTex); _wristEventTex = null; }
        }

        public void EnsureCameras()
        {
            if (playback == null)
            {
#if UNITY_2023_1_OR_NEWER
                playback = FindAnyObjectByType<GhostPlaybackManager>();
#else
                playback = FindObjectOfType<GhostPlaybackManager>();
#endif
            }

            Transform root = playback != null ? playback.transform : transform;

            // 1. Anthropomorphic Eye-Level Global Sensor Camera (positioned at human head relative to shoulder mount, tilted slightly toward the object)
            SPWM.Robotics.ArticulatedRobotRig.GetEyeLevelCameraPose(root.position, out Vector3 eyeCamPos, out Quaternion eyeCamRot);
            if (playback != null && playback.IsLoaded)
            {
                Vector3 lookTarget = playback.InitialGraspTargetWorldPosition;
                if (playback.TargetRig != null && playback.TargetRig.EndEffectorTip != null)
                {
                    lookTarget = Vector3.Lerp(lookTarget, playback.TargetRig.EndEffectorTip.position, 0.25f);
                }
                Vector3 toTarget = (lookTarget - eyeCamPos).normalized;
                if (toTarget.sqrMagnitude > 1e-6f)
                {
                    eyeCamRot = Quaternion.LookRotation(toTarget, Vector3.up);
                }
            }
            int uiMask = ~(1 << 5); // Exclude UI layer (5) from sensory cameras
            if (globalCamera == null)
            {
                var existing = GameObject.Find("PlaybackGlobalSensorCamera");
                if (existing != null)
                {
                    globalCamera = existing.GetComponent<Camera>();
                }
                else
                {
                    var camGo = new GameObject("PlaybackGlobalSensorCamera");
                    camGo.transform.SetParent(root, false);
                    globalCamera = camGo.AddComponent<Camera>();
                    globalCamera.fieldOfView = 60f;
                    globalCamera.nearClipPlane = 0.05f;
                    globalCamera.farClipPlane = 50f;
                    globalCamera.clearFlags = CameraClearFlags.SolidColor;
                    globalCamera.backgroundColor = new Color(0.08f, 0.10f, 0.14f, 1f);
                }
            }
            if (globalCamera != null)
            {
                globalCamera.cullingMask = uiMask;
                globalCamera.transform.SetPositionAndRotation(eyeCamPos, eyeCamRot);
            }

            // 2. Egocentric Wrist Sensor Camera (mounted directly on Target End-Effector)
            Transform wristMount = null;
            if (playback != null && playback.TargetRig != null)
            {
                wristMount = playback.TargetRig.WristCameraMount != null
                    ? playback.TargetRig.WristCameraMount
                    : playback.TargetRig.EndEffectorTip;
            }

            if (wristCamera == null)
            {
                var existing = GameObject.Find("PlaybackWristSensorCamera");
                if (existing != null)
                {
                    wristCamera = existing.GetComponent<Camera>();
                }
                else
                {
                    var camGo = new GameObject("PlaybackWristSensorCamera");
                    wristCamera = camGo.AddComponent<Camera>();
                    wristCamera.fieldOfView = 65f;
                    wristCamera.nearClipPlane = 0.02f;
                    wristCamera.farClipPlane = 25f;
                    wristCamera.clearFlags = CameraClearFlags.SolidColor;
                    wristCamera.backgroundColor = new Color(0.08f, 0.10f, 0.14f, 1f);
                }
            }
            if (wristCamera != null)
            {
                wristCamera.cullingMask = uiMask;
            }

            if (wristMount != null && wristCamera != null && wristCamera.transform.parent != wristMount)
            {
                wristCamera.transform.SetParent(wristMount, false);
                wristCamera.transform.localPosition = Vector3.zero;
                wristCamera.transform.localRotation = Quaternion.identity;
            }

            // Cameras only render manually on demand
            if (globalCamera != null) globalCamera.enabled = false;
            if (wristCamera != null) wristCamera.enabled = false;
        }

        public void OnPlaybackLoaded(GhostPlaybackManager pm)
        {
            EnsureTextures();
            EnsureCameras();
            OnPlaybackRestarted(pm);
        }

        private void OnPlaybackRestarted(GhostPlaybackManager pm)
        {
            _isFirstFrame = true;
            _lastFrameIndex = -1;
            if (_prevGlobalLogL != null) Array.Clear(_prevGlobalLogL, 0, _prevGlobalLogL.Length);
            if (_prevWristLogL != null) Array.Clear(_prevWristLogL, 0, _prevWristLogL.Length);

            RefreshSensors(force: true);
        }

        private void LateUpdate()
        {
            if (playback != null && playback.IsPlaying)
            {
                RefreshSensors(force: false);
            }
        }

        /// <summary>
        /// Captures sensor camera feeds, processes differential event log-luminance frames,
        /// and updates both optical and event stream visual elements on the UI.
        /// </summary>
        public void RefreshSensors(bool force)
        {
            if (playback == null || !playback.IsLoaded) return;

            int currentFrame = Mathf.FloorToInt(playback.FramePosition);
            if (!force && currentFrame == _lastFrameIndex) return;
            _lastFrameIndex = currentFrame;

            EnsureTextures();
            EnsureCameras();

            // Check if trajectory file has pre-recorded events
            GhostFrame frameData = (playback.Data != null && currentFrame < playback.Data.FrameCount)
                ? playback.Data.frames[currentFrame]
                : null;

            bool hasPrerecordedEvents = frameData != null &&
                                       ((frameData.global_events != null && frameData.global_events.Length >= sensorWidth * sensorHeight) ||
                                        (frameData.wrist_events != null && frameData.wrist_events.Length >= sensorWidth * sensorHeight));

            // Hide the holographic ghost so sensory cameras capture physical ground truth only
            playback.SetGhostVisible(false);

            try
            {
                // 1. Render Optical Views (high-res for display)
                if (globalCamera != null && _globalOpticalRT != null)
                {
                    globalCamera.targetTexture = _globalOpticalRT;
                    globalCamera.Render();
                }

                if (wristCamera != null && _wristOpticalRT != null)
                {
                    wristCamera.targetTexture = _wristOpticalRT;
                    wristCamera.Render();
                }

                // 2. Compute Neuromorphic Events
                if (hasPrerecordedEvents)
                {
                    RenderPrerecordedEvents(frameData);
                }
                else
                {
                    ComputeLiveSensoryEvents();
                }
            }
            finally
            {
                // Restore ghost visibility for main camera rendering
                playback.SetGhostVisible(true);
            }

            UpdateUIStatus();
        }

        private void ComputeLiveSensoryEvents()
        {
            int numPixels = sensorWidth * sensorHeight;

            // Global Camera Event Differencing
            if (globalCamera != null && _globalSensorRT != null)
            {
                globalCamera.targetTexture = _globalSensorRT;
                globalCamera.Render();

                RenderTexture.active = _globalSensorRT;
                _globalReadbackTex.ReadPixels(new Rect(0, 0, sensorWidth, sensorHeight), 0, 0);
                _globalReadbackTex.Apply();

                Color32[] rawCols = _globalReadbackTex.GetPixels32();
                int posCount = 0, negCount = 0;

                for (int i = 0; i < numPixels; i++)
                {
                    Color32 c = rawCols[i];
                    // ITU-R BT.709 luminance
                    float lum = (0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b) / 255.0f;
                    float logL = Mathf.Log(Mathf.Max(lum, 0f) + 1e-3f);

                    if (_isFirstFrame)
                    {
                        _prevGlobalLogL[i] = logL;
                        _globalPixels[i] = backgroundColor;
                    }
                    else
                    {
                        float deltaL = logL - _prevGlobalLogL[i];
                        if (deltaL > eventThreshold)
                        {
                            _globalPixels[i] = onSpikeColor;
                            _prevGlobalLogL[i] = logL;
                            posCount++;
                        }
                        else if (deltaL < -eventThreshold)
                        {
                            _globalPixels[i] = offSpikeColor;
                            _prevGlobalLogL[i] = logL;
                            negCount++;
                        }
                        else
                        {
                            _globalPixels[i] = backgroundColor;
                        }
                    }
                }

                _globalEventTex.SetPixels32(_globalPixels);
                _globalEventTex.Apply();
                LastGlobalActivity = (float)(posCount + negCount) / numPixels;
            }

            // Wrist Camera Event Differencing
            if (wristCamera != null && _wristSensorRT != null)
            {
                wristCamera.targetTexture = _wristSensorRT;
                wristCamera.Render();

                RenderTexture.active = _wristSensorRT;
                _wristReadbackTex.ReadPixels(new Rect(0, 0, sensorWidth, sensorHeight), 0, 0);
                _wristReadbackTex.Apply();

                Color32[] rawCols = _wristReadbackTex.GetPixels32();
                int posCount = 0, negCount = 0;

                for (int i = 0; i < numPixels; i++)
                {
                    Color32 c = rawCols[i];
                    float lum = (0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b) / 255.0f;
                    float logL = Mathf.Log(Mathf.Max(lum, 0f) + 1e-3f);

                    if (_isFirstFrame)
                    {
                        _prevWristLogL[i] = logL;
                        _wristPixels[i] = backgroundColor;
                    }
                    else
                    {
                        float deltaL = logL - _prevWristLogL[i];
                        if (deltaL > eventThreshold)
                        {
                            _wristPixels[i] = onSpikeColor;
                            _prevWristLogL[i] = logL;
                            posCount++;
                        }
                        else if (deltaL < -eventThreshold)
                        {
                            _wristPixels[i] = offSpikeColor;
                            _prevWristLogL[i] = logL;
                            negCount++;
                        }
                        else
                        {
                            _wristPixels[i] = backgroundColor;
                        }
                    }
                }

                _wristEventTex.SetPixels32(_wristPixels);
                _wristEventTex.Apply();
                LastWristActivity = (float)(posCount + negCount) / numPixels;
            }

            RenderTexture.active = null;
            _isFirstFrame = false;
        }

        private void RenderPrerecordedEvents(GhostFrame frame)
        {
            int numPixels = sensorWidth * sensorHeight;

            if (frame.global_events != null && frame.global_events.Length >= numPixels)
            {
                bool dualPolarity = frame.global_events.Length >= 2 * numPixels;
                int posCount = 0, negCount = 0;

                for (int i = 0; i < numPixels; i++)
                {
                    float pos = frame.global_events[i];
                    float neg = dualPolarity ? frame.global_events[numPixels + i] : 0f;

                    if (pos > 0.5f)
                    {
                        _globalPixels[i] = onSpikeColor;
                        posCount++;
                    }
                    else if (neg > 0.5f)
                    {
                        _globalPixels[i] = offSpikeColor;
                        negCount++;
                    }
                    else
                    {
                        _globalPixels[i] = backgroundColor;
                    }
                }
                _globalEventTex.SetPixels32(_globalPixels);
                _globalEventTex.Apply();
                LastGlobalActivity = (float)(posCount + negCount) / numPixels;
            }

            if (frame.wrist_events != null && frame.wrist_events.Length >= numPixels)
            {
                bool dualPolarity = frame.wrist_events.Length >= 2 * numPixels;
                int posCount = 0, negCount = 0;

                for (int i = 0; i < numPixels; i++)
                {
                    float pos = frame.wrist_events[i];
                    float neg = dualPolarity ? frame.wrist_events[numPixels + i] : 0f;

                    if (pos > 0.5f)
                    {
                        _wristPixels[i] = onSpikeColor;
                        posCount++;
                    }
                    else if (neg > 0.5f)
                    {
                        _wristPixels[i] = offSpikeColor;
                        negCount++;
                    }
                    else
                    {
                        _wristPixels[i] = backgroundColor;
                    }
                }
                _wristEventTex.SetPixels32(_wristPixels);
                _wristEventTex.Apply();
                LastWristActivity = (float)(posCount + negCount) / numPixels;
            }
        }

        private void UpdateUIStatus()
        {
            bool globalActive = LastGlobalActivity > activityThreshold;
            bool wristActive = LastWristActivity > activityThreshold;

            if (globalStatusText != null)
            {
                string statusColor = globalActive ? "#39D353" : "#FF7B72";
                string statusLabel = globalActive ? "ACTIVE" : "QUIESCENT";
                globalStatusText.text = string.Format(Inv,
                    "act: <b>{0:0.0000}</b>  ·  <color={1}>{2}</color>",
                    LastGlobalActivity, statusColor, statusLabel);
            }

            if (wristStatusText != null)
            {
                string statusColor = wristActive ? "#39D353" : "#FF7B72";
                string statusLabel = wristActive ? "ACTIVE" : "QUIESCENT";
                wristStatusText.text = string.Format(Inv,
                    "act: <b>{0:0.0000}</b>  ·  <color={1}>{2}</color>",
                    LastWristActivity, statusColor, statusLabel);
            }
        }
    }
}
