using System;
using UnityEngine;

namespace SPWM.Robotics
{
    /// <summary>
    /// Neuromorphic Vision Bridge for the 3D Unity Robotic Simulator (SPWM-v0.6.0 Dual-Camera Architecture).
    /// Supports simultaneous static global and egocentric wrist-mounted event cameras, rendering each camera view
    /// into a RenderTexture and either:
    ///   1. Dispatching EventCameraLogDiff.compute on the GPU to yield dense [2, H, W] (or unified [4, H, W]) event frames
    ///      using threshold theta = 0.08, or
    ///   2. Exporting high-FPS grayscale luminance frames ([H, W] per camera, or unified [2, H, W]) for Python's EventCameraSimulator.
    /// </summary>
    [RequireComponent(typeof(Camera))]
    public class NeuromorphicEventBridge : MonoBehaviour
    {
        [Header("Event Sensor Resolution & Threshold")]
        [SerializeField] private int sensorWidth = 32;
        [SerializeField] private int sensorHeight = 32;
        [SerializeField] private float eventThreshold = 0.08f;
        [SerializeField] private ComputeShader eventComputeShader;

        [Header("Secondary Egocentric Wrist Sensor (SPWM-v0.6.0)")]
        [SerializeField] private NeuromorphicEventBridge secondaryWristBridge;

        private Camera _sensorCamera;
        private RenderTexture _frameRenderTexture;
        private RenderTexture _prevLogIntensityTexture;
        private ComputeBuffer _denseEventBuffer;
        private Texture2D _cpuReadbackTexture;
        private float[] _prevCpuLogIntensity;
        private bool _isFirstFrame = true;

        public int SensorWidth => sensorWidth;
        public int SensorHeight => sensorHeight;
        public float EventThreshold => eventThreshold;
        public int SensorChannels => secondaryWristBridge != null ? 4 : 2;
        public NeuromorphicEventBridge SecondaryWristBridge
        {
            get => secondaryWristBridge;
            set => secondaryWristBridge = value;
        }

        private void Awake()
        {
            _sensorCamera = GetComponent<Camera>();
            InitializeBuffers(sensorWidth, sensorHeight);
        }

        public void InitializeBuffers(int width, int height)
        {
            if (width <= 0 || width % 4 != 0 || height <= 0 || height % 4 != 0)
            {
                throw new ArgumentException(
                    $"Sensor dimensions ({width}x{height}) must be positive multiples of 4.");
            }

            sensorWidth = width;
            sensorHeight = height;

            ReleaseBuffers();

            if (_sensorCamera == null)
            {
                _sensorCamera = GetComponent<Camera>();
            }

            _frameRenderTexture = new RenderTexture(sensorWidth, sensorHeight, 24, RenderTextureFormat.ARGB32);
            _frameRenderTexture.Create();

            _prevLogIntensityTexture = new RenderTexture(sensorWidth, sensorHeight, 0, RenderTextureFormat.RFloat)
            {
                enableRandomWrite = true
            };
            _prevLogIntensityTexture.Create();

            _denseEventBuffer = new ComputeBuffer(2 * sensorWidth * sensorHeight, sizeof(float));
            _cpuReadbackTexture = new Texture2D(sensorWidth, sensorHeight, TextureFormat.RGBA32, false);
            _prevCpuLogIntensity = new float[sensorWidth * sensorHeight];
            _isFirstFrame = true;

            if (_sensorCamera != null)
            {
                _sensorCamera.targetTexture = _frameRenderTexture;
            }
        }

        public void ResetSensorBaseline()
        {
            _isFirstFrame = true;
            if (_prevCpuLogIntensity != null)
            {
                Array.Clear(_prevCpuLogIntensity, 0, _prevCpuLogIntensity.Length);
            }
            if (secondaryWristBridge != null && secondaryWristBridge != this)
            {
                secondaryWristBridge.ResetSensorBaseline();
            }
        }

        /// <summary>
        /// Simultaneously captures both the static global camera and the egocentric wrist-mounted camera,
        /// returning a unified 4-channel event stream [4 * H * W] ([pos_global, neg_global, pos_wrist, neg_wrist])
        /// alongside the individual global and wrist grayscale frames ([H * W] each) and unified [2 * H * W] grayscale array.
        /// </summary>
        public (float[] unifiedDenseEvents, float[] globalGrayscale, float[] wristGrayscale, float[] unifiedGrayscale) CaptureDualStep(
            NeuromorphicEventBridge explicitWristBridge = null)
        {
            NeuromorphicEventBridge wrist = explicitWristBridge != null ? explicitWristBridge : secondaryWristBridge;
            var (globalEvents, globalGray) = CaptureStep();
            int numPixels = sensorWidth * sensorHeight;

            float[] wristEvents;
            float[] wristGray;
            if (wrist != null && wrist != this)
            {
                (wristEvents, wristGray) = wrist.CaptureStep();
            }
            else
            {
                wristEvents = new float[2 * numPixels];
                wristGray = new float[numPixels];
                Array.Copy(globalEvents, wristEvents, 2 * numPixels);
                Array.Copy(globalGray, wristGray, numPixels);
            }

            float[] unifiedEvents = new float[4 * numPixels];
            Array.Copy(globalEvents, 0, unifiedEvents, 0, 2 * numPixels);
            Array.Copy(wristEvents, 0, unifiedEvents, 2 * numPixels, 2 * numPixels);

            float[] unifiedGray = new float[2 * numPixels];
            Array.Copy(globalGray, 0, unifiedGray, 0, numPixels);
            Array.Copy(wristGray, 0, unifiedGray, numPixels, numPixels);

            return (unifiedEvents, globalGray, wristGray, unifiedGray);
        }

        /// <summary>
        /// Renders the current camera view and captures both the dense [2, H, W] event frame
        /// and the normalized [H, W] grayscale frame.
        /// </summary>
        public (float[] denseEvents, float[] grayscaleFrame) CaptureStep()
        {
            if (_sensorCamera == null)
            {
                _sensorCamera = GetComponent<Camera>();
                if (_sensorCamera != null && _sensorCamera.targetTexture == null)
                {
                    _sensorCamera.targetTexture = _frameRenderTexture;
                }
            }

            if (_sensorCamera != null)
            {
                _sensorCamera.Render();
            }

            int numPixels = sensorWidth * sensorHeight;
            float[] grayscale = CaptureGrayscaleCpu();
            float[] denseEvents = new float[2 * numPixels];

            if (eventComputeShader != null && SystemInfo.supportsComputeShaders)
            {
                int kernel = eventComputeShader.FindKernel("ComputeLogIntensityEvents");
                eventComputeShader.SetTexture(kernel, "CurrentFrame", _frameRenderTexture);
                eventComputeShader.SetTexture(kernel, "PrevLogIntensity", _prevLogIntensityTexture);
                eventComputeShader.SetBuffer(kernel, "DenseEventBuffer", _denseEventBuffer);
                eventComputeShader.SetInt("Width", sensorWidth);
                eventComputeShader.SetInt("Height", sensorHeight);
                eventComputeShader.SetFloat("EventThreshold", eventThreshold);
                eventComputeShader.SetInt("IsFirstFrame", _isFirstFrame ? 1 : 0);

                int groupsX = Mathf.CeilToInt(sensorWidth / 8.0f);
                int groupsY = Mathf.CeilToInt(sensorHeight / 8.0f);
                eventComputeShader.Dispatch(kernel, groupsX, groupsY, 1);

                _denseEventBuffer.GetData(denseEvents);
                _isFirstFrame = false;
            }
            else
            {
                // CPU computation matching EventCameraLogDiff.compute and Python EventCameraSimulator
                for (int i = 0; i < numPixels; i++)
                {
                    float logL = grayscale[i];
                    if (_isFirstFrame)
                    {
                        _prevCpuLogIntensity[i] = logL;
                        denseEvents[i] = 0.0f;
                        denseEvents[numPixels + i] = 0.0f;
                    }
                    else
                    {
                        float deltaL = logL - _prevCpuLogIntensity[i];
                        if (deltaL > eventThreshold)
                        {
                            denseEvents[i] = 1.0f;
                            denseEvents[numPixels + i] = 0.0f;
                            _prevCpuLogIntensity[i] = logL;
                        }
                        else if (deltaL < -eventThreshold)
                        {
                            denseEvents[i] = 0.0f;
                            denseEvents[numPixels + i] = 1.0f;
                            _prevCpuLogIntensity[i] = logL;
                        }
                        else
                        {
                            denseEvents[i] = 0.0f;
                            denseEvents[numPixels + i] = 0.0f;
                        }
                    }
                }
                _isFirstFrame = false;
            }

            return (denseEvents, grayscale);
        }

        private float[] CaptureGrayscaleCpu()
        {
            RenderTexture previous = RenderTexture.active;
            RenderTexture.active = _frameRenderTexture;
            _cpuReadbackTexture.ReadPixels(new Rect(0, 0, sensorWidth, sensorHeight), 0, 0);
            _cpuReadbackTexture.Apply();
            RenderTexture.active = previous;

            Color[] pixels = _cpuReadbackTexture.GetPixels();
            float[] grayscale = new float[pixels.Length];
            for (int i = 0; i < pixels.Length; i++)
            {
                Color c = pixels[i];
                grayscale[i] = Mathf.Clamp01(0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b);
            }
            return grayscale;
        }

        private void ReleaseBuffers()
        {
            if (_frameRenderTexture != null)
            {
                _frameRenderTexture.Release();
                DestroyImmediate(_frameRenderTexture);
            }
            if (_prevLogIntensityTexture != null)
            {
                _prevLogIntensityTexture.Release();
                DestroyImmediate(_prevLogIntensityTexture);
            }
            if (_denseEventBuffer != null)
            {
                _denseEventBuffer.Release();
                _denseEventBuffer = null;
            }
        }

        private void OnDestroy()
        {
            ReleaseBuffers();
        }
    }
}
