using System;
using UnityEngine;

namespace SPWM.Robotics
{
    /// <summary>
    /// Neuromorphic Vision Bridge for the 3D Unity Robotic Simulator.
    /// Renders the scene camera view into a RenderTexture and either:
    ///   1. Dispatches EventCameraLogDiff.compute on the GPU to yield dense [2, H, W] event frames
    ///      using threshold theta = 0.08, or
    ///   2. Exports high-FPS grayscale luminance frames [H, W] for Python's EventCameraSimulator.
    /// </summary>
    [RequireComponent(typeof(Camera))]
    public class NeuromorphicEventBridge : MonoBehaviour
    {
        [Header("Event Sensor Resolution & Threshold")]
        [SerializeField] private int sensorWidth = 32;
        [SerializeField] private int sensorHeight = 32;
        [SerializeField] private float eventThreshold = 0.08f;
        [SerializeField] private ComputeShader eventComputeShader;

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

            _frameRenderTexture = new RenderTexture(sensorWidth, sensorHeight, 24, RenderTextureFormat.ARGBFloat);
            _frameRenderTexture.Create();

            _prevLogIntensityTexture = new RenderTexture(sensorWidth, sensorHeight, 0, RenderTextureFormat.RFloat)
            {
                enableRandomWrite = true
            };
            _prevLogIntensityTexture.Create();

            _denseEventBuffer = new ComputeBuffer(2 * sensorWidth * sensorHeight, sizeof(float));
            _cpuReadbackTexture = new Texture2D(sensorWidth, sensorHeight, TextureFormat.RGBAFloat, false);
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
        }

        /// <summary>
        /// Renders the current camera view and captures both the dense [2, H, W] event frame
        /// and the normalized [H, W] grayscale frame.
        /// </summary>
        public (float[] denseEvents, float[] grayscaleFrame) CaptureStep()
        {
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
                // CPU fallback matching EventCameraLogDiff.compute and Python EventCameraSimulator
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
                        _prevCpuLogIntensity[i] = logL;
                        denseEvents[i] = (deltaL > eventThreshold) ? 1.0f : 0.0f;
                        denseEvents[numPixels + i] = (deltaL < -eventThreshold) ? 1.0f : 0.0f;
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
