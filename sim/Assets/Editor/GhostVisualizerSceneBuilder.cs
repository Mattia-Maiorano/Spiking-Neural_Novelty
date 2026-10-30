#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.UI;
using SPWM.Visualization;

namespace SPWM.Editor
{
    public static class GhostVisualizerSceneBuilder
    {
        private const string SceneSavePath = "Assets/Scenes/GhostVisualizer_Setup.unity";

        [MenuItem("SPWM/Build Visualization Scene", priority = 10)]
        public static void BuildScene()
        {
            // 1. Nuova scena vuota
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            // 2. Luce Direzionale
            var lightGo = new GameObject("Directional Light");
            var light = lightGo.AddComponent<Light>();
            light.type = LightType.Directional;
            light.color = Color.white;
            light.intensity = 1.15f;
            light.shadows = LightShadows.Soft;
            lightGo.transform.rotation = Quaternion.Euler(50f, -30f, 0f);

            // 3. Playback Manager
            var playbackGo = new GameObject("PlaybackManager");
            playbackGo.transform.position = Vector3.zero;
            var playbackManager = playbackGo.AddComponent<GhostPlaybackManager>();

            // 4. Main Camera + Cinematic Tracker
            var cameraGo = new GameObject("Main Camera");
            cameraGo.tag = "MainCamera";
            var cam = cameraGo.AddComponent<Camera>();
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.06f, 0.08f, 0.11f, 1f); // Dark Charcoal/Slate
            cam.fieldOfView = 50f;
            cam.nearClipPlane = 0.05f;
            cam.farClipPlane = 100f;
            cameraGo.AddComponent<AudioListener>();

            var cameraTracker = cameraGo.AddComponent<CinematicCameraTracker>();
            var camSo = new SerializedObject(cameraTracker);
            camSo.FindProperty("playback").objectReferenceValue = playbackManager;
            camSo.FindProperty("elevationDeg").floatValue = 30f;
            camSo.FindProperty("azimuthDeg").floatValue = 45f;
            camSo.FindProperty("screenComposition").vector2Value = new Vector2(0.12f, 0.05f);
            camSo.ApplyModifiedPropertiesWithoutUndo();

            // 5. Canvas e EventSystem
            var canvasGo = new GameObject("TelemetryCanvas");
            var canvas = canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            var scaler = canvasGo.AddComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1080, 1920); // Mobile / LinkedIn Portrait
            scaler.matchWidthOrHeight = 0.5f;
            canvasGo.AddComponent<GraphicRaycaster>();

            var eventSystemGo = new GameObject("EventSystem");
            eventSystemGo.AddComponent<UnityEngine.EventSystems.EventSystem>();
            eventSystemGo.AddComponent<UnityEngine.EventSystems.StandaloneInputModule>();

            // 6. Costruzione Gerarchia HUD Panel (Bottom-Left)
            var hudPanelGo = CreateUIElement("HUD_Panel", canvasGo.transform);
            var hudPanelImg = hudPanelGo.AddComponent<Image>();
            hudPanelImg.color = new Color(0.051f, 0.067f, 0.090f, 0.90f); // #0D1117E6

            var hudRect = hudPanelGo.GetComponent<RectTransform>();
            hudRect.anchorMin = new Vector2(0f, 0f);
            hudRect.anchorMax = new Vector2(0f, 0f);
            hudRect.pivot = new Vector2(0f, 0f);
            hudRect.anchoredPosition = new Vector2(40f, 40f);
            hudRect.sizeDelta = new Vector2(420f, 480f);

            var font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");

            // Header Title & Subtitle
            var titleText = CreateText("Header_Title", hudPanelGo.transform, "SPWM v0.4.0 · GHOST EXECUTION", font, 24, FontStyle.Bold, Color.white);
            PositionRect(titleText.rectTransform, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, 1f), new Vector2(20f, -20f), new Vector2(-40f, 32f));

            var subText = CreateText("Header_Subtitle", hudPanelGo.transform, "M1 autoregressive rollout · H=100 · Δt=50ms", font, 14, FontStyle.Normal, new Color(0.55f, 0.58f, 0.62f));
            PositionRect(subText.rectTransform, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, 1f), new Vector2(20f, -54f), new Vector2(-40f, 22f));

            // Horizon Progress Group
            var stepText = CreateText("StepCounter", hudPanelGo.transform, "1 / 100", font, 32, FontStyle.Bold, Color.white);
            PositionRect(stepText.rectTransform, new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(20f, -92f), new Vector2(200f, 38f));

            var timeText = CreateText("TimeLabel", hudPanelGo.transform, "t = 0.05 s", font, 16, FontStyle.Normal, new Color(0.55f, 0.58f, 0.62f));
            PositionRect(timeText.rectTransform, new Vector2(1f, 1f), new Vector2(1f, 1f), new Vector2(1f, 1f), new Vector2(-20f, -100f), new Vector2(160f, 26f));
            timeText.alignment = TextAnchor.MiddleRight;

            var progressTrack = CreateUIElement("ProgressBar_Track", hudPanelGo.transform);
            var trackImg = progressTrack.AddComponent<Image>();
            trackImg.color = new Color(0.13f, 0.15f, 0.18f); // #21262D
            PositionRect(progressTrack.GetComponent<RectTransform>(), new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, 1f), new Vector2(20f, -136f), new Vector2(-40f, 6f));

            var progressFill = CreateUIElement("ProgressBar_Fill", progressTrack.transform);
            var fillImg = progressFill.AddComponent<Image>();
            fillImg.color = new Color(0f, 0.85f, 1f); // #00D9FF
            var fillRect = progressFill.GetComponent<RectTransform>();
            fillRect.anchorMin = new Vector2(0f, 0f);
            fillRect.anchorMax = new Vector2(0f, 1f);
            fillRect.pivot = new Vector2(0f, 0.5f);
            fillRect.anchoredPosition = Vector2.zero;
            fillRect.sizeDelta = new Vector2(0f, 0f);

            // Attractor Convergence Group (Sparkline)
            var convLabel = CreateText("Conv_Label", hudPanelGo.transform, "ATTRACTOR CONVERGENCE", font, 13, FontStyle.Bold, new Color(0.35f, 0.65f, 1f));
            PositionRect(convLabel.rectTransform, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, 1f), new Vector2(20f, -165f), new Vector2(-40f, 20f));

            var velMetricText = CreateText("VelocityMetric", hudPanelGo.transform, "×1.00", font, 28, FontStyle.Bold, new Color(0.22f, 0.83f, 0.33f)); // #39D353
            PositionRect(velMetricText.rectTransform, new Vector2(1f, 1f), new Vector2(1f, 1f), new Vector2(1f, 1f), new Vector2(-20f, -188f), new Vector2(120f, 34f));
            velMetricText.alignment = TextAnchor.MiddleRight;

            var sparklineGo = CreateUIElement("Sparkline", hudPanelGo.transform);
            var sparkline = sparklineGo.AddComponent<UISparkline>();
            PositionRect(sparklineGo.GetComponent<RectTransform>(), new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, 1f), new Vector2(20f, -225f), new Vector2(-40f, 80f));

            // Phase Shift Group (Position Error)
            var phaseLabel = CreateText("Phase_Label", hudPanelGo.transform, "PHASE SHIFT (POSITION DRIFT)", font, 13, FontStyle.Bold, new Color(1f, 0.65f, 0.34f));
            PositionRect(phaseLabel.rectTransform, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, 1f), new Vector2(36f, -325f), new Vector2(-56f, 20f));

            var accentBarGo = CreateUIElement("AccentBar", hudPanelGo.transform);
            var accentBarImg = accentBarGo.AddComponent<Image>();
            accentBarImg.color = new Color(0.22f, 0.83f, 0.33f);
            PositionRect(accentBarGo.GetComponent<RectTransform>(), new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(20f, -325f), new Vector2(6f, 75f));

            var posErrText = CreateText("PositionErrorText", hudPanelGo.transform, "0.175 rad", font, 28, FontStyle.Bold, Color.white);
            PositionRect(posErrText.rectTransform, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, 1f), new Vector2(36f, -350f), new Vector2(-56f, 34f));

            var posErrDegText = CreateText("PositionErrorDegText", hudPanelGo.transform, "≈ 10.1° joint-space drift", font, 15, FontStyle.Normal, new Color(0.55f, 0.58f, 0.62f));
            PositionRect(posErrDegText.rectTransform, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, 1f), new Vector2(36f, -384f), new Vector2(-56f, 22f));

            // 7. Wiring Componenti su TelemetryDisplay
            var telemetry = canvasGo.AddComponent<TelemetryDisplay>();
            var telemSo = new SerializedObject(telemetry);
            telemSo.FindProperty("playback").objectReferenceValue = playbackManager;
            telemSo.FindProperty("titleText").objectReferenceValue = titleText;
            telemSo.FindProperty("subtitleText").objectReferenceValue = subText;
            telemSo.FindProperty("stepCounterText").objectReferenceValue = stepText;
            telemSo.FindProperty("timeText").objectReferenceValue = timeText;
            telemSo.FindProperty("progressFill").objectReferenceValue = fillRect;
            telemSo.FindProperty("velocitySparkline").objectReferenceValue = sparkline;
            telemSo.FindProperty("velocityValueText").objectReferenceValue = velMetricText;
            telemSo.FindProperty("positionErrorText").objectReferenceValue = posErrText;
            telemSo.FindProperty("positionErrorDegText").objectReferenceValue = posErrDegText;
            telemSo.FindProperty("positionErrorAccent").objectReferenceValue = accentBarImg;
            telemSo.ApplyModifiedPropertiesWithoutUndo();

            // 8. Salvataggio scena
            if (!AssetDatabase.IsValidFolder("Assets/Scenes"))
            {
                AssetDatabase.CreateFolder("Assets", "Scenes");
            }

            EditorSceneManager.SaveScene(scene, SceneSavePath);
            AssetDatabase.Refresh();
            Debug.Log($"[SPWM] Scena di visualizzazione creata con successo in: {SceneSavePath}");
        }

        private static GameObject CreateUIElement(string name, Transform parent)
        {
            var go = new GameObject(name, typeof(RectTransform));
            go.transform.SetParent(parent, false);
            return go;
        }

        private static Text CreateText(string name, Transform parent, string content, Font font, int size, FontStyle style, Color color)
        {
            var go = CreateUIElement(name, parent);
            var text = go.AddComponent<Text>();
            text.text = content;
            text.font = font;
            text.fontSize = size;
            text.fontStyle = style;
            text.color = color;
            text.alignment = TextAnchor.MiddleLeft;
            text.raycastTarget = false;
            return text;
        }

        private static void PositionRect(RectTransform rect, Vector2 anchorMin, Vector2 anchorMax, Vector2 pivot, Vector2 pos, Vector2 size)
        {
            rect.anchorMin = anchorMin;
            rect.anchorMax = anchorMax;
            rect.pivot = pivot;
            rect.anchoredPosition = pos;
            rect.sizeDelta = size;
        }
        [MenuItem("SPWM/Export Dataset from Scene", priority = 20)]
        public static void ExportDatasetFromScene()
        {
            var logger = EnsureSimulationRig();
            logger.ExportBatchTrajectories();
        }

        /// <summary>
        /// Entry point for headless command-line execution:
        /// /Applications/Unity/Hub/Editor/6000.5.7f1/Unity.app/Contents/MacOS/Unity -batchmode -projectPath ... -executeMethod SPWM.Editor.GhostVisualizerSceneBuilder.HeadlessBatchExport -quit
        /// </summary>
        public static void HeadlessBatchExport()
        {
            Debug.Log("[SPWM.HeadlessBatchExport] Initializing headless physics simulation environment...");
            var logger = EnsureSimulationRig();

            // Check if count or output dir passed via CLI args, e.g.: -totalTrajectories 4000 -exportDir ExportedTrajectories
            string[] args = System.Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "-totalTrajectories" && i < args.Length - 1 && int.TryParse(args[i + 1], out int count))
                {
                    logger.TotalTrajectoriesToExport = count;
                }
                else if (args[i] == "-exportDir" && i < args.Length - 1)
                {
                    logger.ExportDirectory = args[i + 1];
                }
                else if (args[i] == "-compressGzip" && i < args.Length - 1 && bool.TryParse(args[i + 1], out bool gz))
                {
                    logger.CompressGzip = gz;
                }
                else if (args[i] == "-includeDenseEvents" && i < args.Length - 1 && bool.TryParse(args[i + 1], out bool inc))
                {
                    logger.IncludeDenseEvents = inc;
                }
            }

            Debug.Log($"[SPWM.HeadlessBatchExport] Starting batch export of {logger.TotalTrajectoriesToExport} trajectories into '{logger.ExportDirectory}'...");
            logger.ExportBatchTrajectories();
            Debug.Log("[SPWM.HeadlessBatchExport] Finished batch export successfully.");
        }

        /// <summary>
        /// Renders camera frames directly from Unity's scene for 100% fidelity video generation:
        /// Unity -batchmode -projectPath ... -executeMethod SPWM.Editor.GhostVisualizerSceneBuilder.HeadlessRenderVideo -framesDir <dir> -trajectoryJson <path> -width 1280 -height 720 -quit
        /// </summary>
        public static void HeadlessRenderVideo()
        {
            Debug.Log("[SPWM.HeadlessRenderVideo] Starting Unity simulation direct frame capture...");
            string framesDir = "results/videos/unity_frames";
            string trajJsonPath = "sim/Assets/StreamingAssets/trajectory_ghost.json";
            int width = 1280;
            int height = 720;

            string[] args = System.Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length - 1; i++)
            {
                if (args[i] == "-framesDir") framesDir = args[i + 1];
                else if (args[i] == "-trajectoryJson") trajJsonPath = args[i + 1];
                else if (args[i] == "-width" && int.TryParse(args[i + 1], out int w)) width = w;
                else if (args[i] == "-height" && int.TryParse(args[i + 1], out int h)) height = h;
            }

            if (!System.IO.File.Exists(trajJsonPath))
            {
                Debug.LogError($"[SPWM.HeadlessRenderVideo] Trajectory file not found: {trajJsonPath}");
                return;
            }

            System.IO.Directory.CreateDirectory(framesDir);

            // Load visualization scene
            if (System.IO.File.Exists(SceneSavePath))
            {
                EditorSceneManager.OpenScene(SceneSavePath, OpenSceneMode.Single);
            }
            else
            {
                BuildScene();
            }

            var playback = Object.FindFirstObjectByType<GhostPlaybackManager>();
            if (playback == null)
            {
                Debug.LogError("[SPWM.HeadlessRenderVideo] GhostPlaybackManager not found in scene!");
                return;
            }

            playback.LoadTrajectory(trajJsonPath);

            var camObj = GameObject.FindWithTag("MainCamera");
            if (camObj == null) camObj = GameObject.Find("Main Camera");
            var cam = camObj != null ? camObj.GetComponent<Camera>() : Camera.main;
            if (cam == null)
            {
                Debug.LogError("[SPWM.HeadlessRenderVideo] Camera not found!");
                return;
            }

            // Set up RenderTexture for direct camera grab
            RenderTexture rt = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32);
            rt.antiAliasing = 4;
            cam.targetTexture = rt;
            Texture2D tex = new Texture2D(width, height, TextureFormat.RGB24, false);

            int totalFrames = playback.FrameCount;
            var tracker = camObj != null ? camObj.GetComponent<CinematicCameraTracker>() : null;
            var telemetry = Object.FindFirstObjectByType<TelemetryDisplay>();

            // Warm up camera tracking
            var lateUpdateMethod = typeof(CinematicCameraTracker).GetMethod("LateUpdate", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var refreshMethod = typeof(TelemetryDisplay).GetMethod("Refresh", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var applyPoseMethod = typeof(GhostPlaybackManager).GetMethod("ApplyPose", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

            Debug.Log($"[SPWM.HeadlessRenderVideo] Capturing {totalFrames} frames ({width}x{height}) to {framesDir}...");

            for (int f = 0; f < totalFrames; f++)
            {
                if (applyPoseMethod != null)
                {
                    applyPoseMethod.Invoke(playback, new object[] { (float)f });
                }

                if (tracker != null && lateUpdateMethod != null)
                {
                    lateUpdateMethod.Invoke(tracker, null);
                }

                if (telemetry != null && refreshMethod != null)
                {
                    refreshMethod.Invoke(telemetry, new object[] { true });
                }

                RenderTexture.active = rt;
                cam.Render();
                tex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
                tex.Apply();

                byte[] pngBytes = tex.EncodeToPNG();
                string framePath = System.IO.Path.Combine(framesDir, $"frame_{f:D5}.png");
                System.IO.File.WriteAllBytes(framePath, pngBytes);
            }

            RenderTexture.active = null;
            cam.targetTexture = null;
            Object.DestroyImmediate(rt);
            Object.DestroyImmediate(tex);

            Debug.Log($"[SPWM.HeadlessRenderVideo] Successfully rendered {totalFrames} frames to {framesDir}.");
        }

        public static SPWM.Robotics.RoboticTelemetryLogger EnsureSimulationRig()
        {
            var logger = Object.FindFirstObjectByType<SPWM.Robotics.RoboticTelemetryLogger>();
            if (logger != null) return logger;

            // Build deterministic physics simulation rig in scene if not already present
            var rootGo = new GameObject("SimulationRig_Physics");
            var robotRig = rootGo.AddComponent<SPWM.Robotics.ArticulatedRobotRig>();
            robotRig.BuildProceduralRig(SPWM.Robotics.RobotArchetype.ManipulatorArm);

            var spawner = rootGo.AddComponent<SPWM.Robotics.ProceduralObjectSpawner>();
            var controller = rootGo.AddComponent<SPWM.Robotics.ActionConditionedController>();

            // 1. Static Global Sensor Camera with NeuromorphicEventBridge
            var camGo = new GameObject("NeuromorphicCamera");
            camGo.transform.SetParent(rootGo.transform, false);
            camGo.transform.localPosition = new Vector3(0.40f, 0.45f, 0.55f);
            camGo.transform.LookAt(new Vector3(0.20f, 0.15f, 0.10f));
            var cam = camGo.AddComponent<Camera>();
            cam.fieldOfView = 50f;
            cam.nearClipPlane = 0.05f;
            cam.farClipPlane = 50f;
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.08f, 0.10f, 0.14f, 1f);

            // 2. Egocentric Wrist-Mounted Sensor Camera attached directly to the end-effector (SPWM-v0.6.0)
            var wristCamGo = new GameObject("WristNeuromorphicCamera");
            wristCamGo.transform.SetParent(robotRig.WristCameraMountTransform, false);
            wristCamGo.transform.localPosition = Vector3.zero;
            wristCamGo.transform.localRotation = Quaternion.identity;
            var wristCam = wristCamGo.AddComponent<Camera>();
            wristCam.fieldOfView = 65f;
            wristCam.nearClipPlane = 0.02f;
            wristCam.farClipPlane = 25f;
            wristCam.clearFlags = CameraClearFlags.SolidColor;
            wristCam.backgroundColor = new Color(0.08f, 0.10f, 0.14f, 1f);

            // Directional Light for 3D visual perception
            var lightGo = new GameObject("SensorDirectionalLight");
            lightGo.transform.SetParent(rootGo.transform, false);
            var light = lightGo.AddComponent<Light>();
            light.type = LightType.Directional;
            light.color = Color.white;
            light.intensity = 1.25f;
            lightGo.transform.rotation = Quaternion.Euler(50f, -30f, 0f);

            var eventBridge = camGo.AddComponent<SPWM.Robotics.NeuromorphicEventBridge>();
            eventBridge.InitializeBuffers(32, 32);

            var wristEventBridge = wristCamGo.AddComponent<SPWM.Robotics.NeuromorphicEventBridge>();
            wristEventBridge.InitializeBuffers(32, 32);
            eventBridge.SecondaryWristBridge = wristEventBridge;

            logger = rootGo.AddComponent<SPWM.Robotics.RoboticTelemetryLogger>();
            var so = new SerializedObject(logger);
            so.FindProperty("robotRig").objectReferenceValue = robotRig;
            so.FindProperty("controller").objectReferenceValue = controller;
            so.FindProperty("objectSpawner").objectReferenceValue = spawner;
            so.FindProperty("eventBridge").objectReferenceValue = eventBridge;
            so.FindProperty("wristEventBridge").objectReferenceValue = wristEventBridge;
            so.FindProperty("exportDirectory").stringValue = "ExportedTrajectories";
            so.FindProperty("totalTrajectoriesToExport").intValue = 4000;
            so.FindProperty("sequenceLength").intValue = 105;
            so.FindProperty("physicsDt").floatValue = 0.05f;
            so.ApplyModifiedPropertiesWithoutUndo();

            return logger;
        }
    }
}
#endif

