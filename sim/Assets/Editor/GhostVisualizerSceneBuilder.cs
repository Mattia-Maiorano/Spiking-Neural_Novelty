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

            // 4. Main Camera + Anthropomorphic Eye-Level Tracker
            var cameraGo = new GameObject("Main Camera");
            cameraGo.tag = "MainCamera";
            SPWM.Robotics.ArticulatedRobotRig.GetEyeLevelCameraPose(Vector3.zero, out Vector3 mainEyePos, out Quaternion mainEyeRot);
            cameraGo.transform.SetPositionAndRotation(mainEyePos, mainEyeRot);
            var cam = cameraGo.AddComponent<Camera>();
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.06f, 0.08f, 0.11f, 1f); // Dark Charcoal/Slate
            cam.fieldOfView = 60f;
            cam.nearClipPlane = 0.05f;
            cam.farClipPlane = 100f;
            cameraGo.AddComponent<AudioListener>();

            var cameraTracker = cameraGo.AddComponent<CinematicCameraTracker>();
            var camSo = new SerializedObject(cameraTracker);
            camSo.FindProperty("playback").objectReferenceValue = playbackManager;
            camSo.FindProperty("useAnthropomorphicEyeLevel").boolValue = true;
            camSo.FindProperty("headOffsetFromShoulder").vector3Value = SPWM.Robotics.ArticulatedRobotRig.HeadOffsetFromShoulder;
            camSo.FindProperty("forwardWorkspaceLookPoint").vector3Value = SPWM.Robotics.ArticulatedRobotRig.EyeCameraLookTargetOffset;
            camSo.FindProperty("elevationDeg").floatValue = 15f;
            camSo.FindProperty("azimuthDeg").floatValue = 0f;
            camSo.FindProperty("screenComposition").vector2Value = Vector2.zero;
            camSo.ApplyModifiedPropertiesWithoutUndo();

            // 5. Canvas e EventSystem 
            var canvasGo = new GameObject("TelemetryCanvas");
            canvasGo.layer = 5; // UI layer
            var canvas = canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = 100;
            var scaler = canvasGo.AddComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1920, 1080); // Standard Full HD
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
            hudRect.sizeDelta = new Vector2(440f, 540f);

            var font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");

            // 2.1 Header Title 
            var titleText = CreateText("Header_Title", hudPanelGo.transform, "Spiking Predictor World Model", font, 20, FontStyle.Bold, Color.white);
            PositionRect(titleText.rectTransform, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, 1f), new Vector2(20f, -16f), new Vector2(-20f, 28f));

            // 2.2 Frame and Phase 
            var frameAndPhaseText = CreateText("Frame_And_Phase", hudPanelGo.transform, "1 - burn in", font, 16, FontStyle.Bold, new Color(0.22f, 0.74f, 0.97f));
            PositionRect(frameAndPhaseText.rectTransform, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, 1f), new Vector2(20f, -50f), new Vector2(-40f, 22f));

            // 2.3 Latent Error Sparkline
            var latentLabel = CreateText("Latent_Label", hudPanelGo.transform, "LATENT ERROR", font, 11, FontStyle.Bold, new Color(0.35f, 0.65f, 1f));
            PositionRect(latentLabel.rectTransform, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, 1f), new Vector2(20f, -100f), new Vector2(-40f, 20f));

            var latentSparklineGo = CreateUIElement("Latent_Sparkline", hudPanelGo.transform);
            var latentSparkline = latentSparklineGo.AddComponent<UISparkline>();
            PositionRect(latentSparklineGo.GetComponent<RectTransform>(), new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, 1f), new Vector2(20f, -130f), new Vector2(-40f, 75f));

            // 2.4 Position Error Sparkline
            var posLabel = CreateText("Pos_Label", hudPanelGo.transform, "POSITION ERROR", font, 11, FontStyle.Bold, new Color(1f, 0.65f, 0.34f));
            PositionRect(posLabel.rectTransform, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, 1f), new Vector2(20f, -240f), new Vector2(-40f, 20f));

            var posSparklineGo = CreateUIElement("Position_Sparkline", hudPanelGo.transform);
            var posSparkline = posSparklineGo.AddComponent<UISparkline>();
            PositionRect(posSparklineGo.GetComponent<RectTransform>(), new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, 1f), new Vector2(20f, -270f), new Vector2(-40f, 75f));

            // 2.5 Velocity Error Sparkline
            var velLabel = CreateText("Vel_Label", hudPanelGo.transform, "VELOCITY ERROR", font, 11, FontStyle.Bold, new Color(0.65f, 0.55f, 0.98f));
            PositionRect(velLabel.rectTransform, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, 1f), new Vector2(20f, -380f), new Vector2(-40f, 20f));

            var velSparklineGo = CreateUIElement("Velocity_Sparkline", hudPanelGo.transform);
            var velSparkline = velSparklineGo.AddComponent<UISparkline>();
            PositionRect(velSparklineGo.GetComponent<RectTransform>(), new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, 1f), new Vector2(20f, -410f), new Vector2(-40f, 75f));

            // 7. Wiring Componenti su TelemetryDisplay
            var telemetry = canvasGo.AddComponent<TelemetryDisplay>();
            var telemSo = new SerializedObject(telemetry);
            telemSo.FindProperty("playback").objectReferenceValue = playbackManager;
            telemSo.FindProperty("titleText").objectReferenceValue = titleText;
            telemSo.FindProperty("frameAndPhaseText").objectReferenceValue = frameAndPhaseText;
            telemSo.FindProperty("latentErrorSparkline").objectReferenceValue = latentSparkline;
            telemSo.FindProperty("positionErrorSparkline").objectReferenceValue = posSparkline;
            telemSo.FindProperty("velocityErrorSparkline").objectReferenceValue = velSparkline;
            telemSo.ApplyModifiedPropertiesWithoutUndo();

            // 8. Costruzione Gerarchia Sensory Streams Panel (Top-Right)
            var sensoryPanelGo = CreateUIElement("Sensory_Panel", canvasGo.transform);
            var sensoryPanelImg = sensoryPanelGo.AddComponent<Image>();
            sensoryPanelImg.color = new Color(0.051f, 0.067f, 0.090f, 0.90f);

            var sensoryRect = sensoryPanelGo.GetComponent<RectTransform>();
            sensoryRect.anchorMin = new Vector2(1f, 1f);
            sensoryRect.anchorMax = new Vector2(1f, 1f);
            sensoryRect.pivot = new Vector2(1f, 1f);
            sensoryRect.anchoredPosition = new Vector2(-40f, -40f);
            sensoryRect.sizeDelta = new Vector2(430f, 560f);

            // Top accent bar
            var sensoryAccentGo = CreateUIElement("Sensory_TopAccent", sensoryPanelGo.transform);
            var sensoryAccentImg = sensoryAccentGo.AddComponent<Image>();
            sensoryAccentImg.color = new Color(0f, 0.85f, 1f); // #00D9FF
            PositionRect(sensoryAccentGo.GetComponent<RectTransform>(), new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, 1f), new Vector2(0f, 0f), new Vector2(0f, 4f));

            // Header Title & Subtitle
            var sensoryTitleText = CreateText("Sensory_Title", sensoryPanelGo.transform, "SENSORY EVENT STREAMS", font, 18, FontStyle.Bold, new Color(0f, 0.85f, 1f));
            PositionRect(sensoryTitleText.rectTransform, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, 1f), new Vector2(18f, -16f), new Vector2(-36f, 26f));

            var sensorySubText = CreateText("Sensory_Subtitle", sensoryPanelGo.transform, "Dual-Camera Neuromorphic Perception (32×32)", font, 12, FontStyle.Normal, new Color(0.55f, 0.58f, 0.62f));
            PositionRect(sensorySubText.rectTransform, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, 1f), new Vector2(18f, -42f), new Vector2(-36f, 18f));

            // Section 1: Global Sensor Camera (Anthropomorphic Eye-Level Head Perspective)
            var globalSectionText = CreateText("Global_Label", sensoryPanelGo.transform, "EYE-LEVEL HEAD CAMERA (WORKSPACE)", font, 12, FontStyle.Bold, new Color(0.35f, 0.65f, 1f));
            PositionRect(globalSectionText.rectTransform, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, 1f), new Vector2(18f, -70f), new Vector2(-36f, 18f));

            // Global Optical Viewport Container
            var globalOptBox = CreateUIElement("Global_Optical_Box", sensoryPanelGo.transform);
            var globalOptBoxImg = globalOptBox.AddComponent<Image>();
            globalOptBoxImg.color = new Color(0.043f, 0.059f, 0.082f);
            PositionRect(globalOptBox.GetComponent<RectTransform>(), new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(18f, -94f), new Vector2(190f, 150f));

            var globalOptRaw = CreateRawImage("Global_Optical_Img", globalOptBox.transform, Color.white);
            PositionRect(globalOptRaw.rectTransform, new Vector2(0f, 0f), new Vector2(1f, 1f), new Vector2(0.5f, 0.5f), Vector2.zero, Vector2.zero);

            var globalOptBadge = CreateText("Global_Opt_Badge", globalOptBox.transform, "OPTICAL FEED", font, 10, FontStyle.Bold, new Color(1f, 1f, 1f, 0.85f));
            PositionRect(globalOptBadge.rectTransform, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, 1f), new Vector2(8f, -6f), new Vector2(-16f, 16f));

            // Global Events Viewport Container
            var globalEvtBox = CreateUIElement("Global_Events_Box", sensoryPanelGo.transform);
            var globalEvtBoxImg = globalEvtBox.AddComponent<Image>();
            globalEvtBoxImg.color = new Color(0.043f, 0.059f, 0.082f);
            PositionRect(globalEvtBox.GetComponent<RectTransform>(), new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(222f, -94f), new Vector2(190f, 150f));

            var globalEvtRaw = CreateRawImage("Global_Events_Img", globalEvtBox.transform, Color.white);
            PositionRect(globalEvtRaw.rectTransform, new Vector2(0f, 0f), new Vector2(1f, 1f), new Vector2(0.5f, 0.5f), Vector2.zero, Vector2.zero);

            var globalEvtBadge = CreateText("Global_Evt_Badge", globalEvtBox.transform, "EVENT STREAM (+/-)", font, 10, FontStyle.Bold, new Color(0f, 0.85f, 1f, 0.95f));
            PositionRect(globalEvtBadge.rectTransform, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, 1f), new Vector2(8f, -6f), new Vector2(-16f, 16f));

            var globalStatusText = CreateText("Global_Status", sensoryPanelGo.transform, "act: 0.0125  ·  ACTIVE", font, 12, FontStyle.Bold, new Color(0.22f, 0.83f, 0.33f));
            PositionRect(globalStatusText.rectTransform, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, 1f), new Vector2(18f, -250f), new Vector2(-36f, 20f));

            // Section 2: Egocentric Wrist Sensor Camera
            var wristSectionText = CreateText("Wrist_Label", sensoryPanelGo.transform, "EGOCENTRIC WRIST CAMERA (GRIPPER)", font, 12, FontStyle.Bold, new Color(0.49f, 0.91f, 0.53f));
            PositionRect(wristSectionText.rectTransform, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, 1f), new Vector2(18f, -276f), new Vector2(-36f, 18f));

            // Wrist Optical Viewport Container
            var wristOptBox = CreateUIElement("Wrist_Optical_Box", sensoryPanelGo.transform);
            var wristOptBoxImg = wristOptBox.AddComponent<Image>();
            wristOptBoxImg.color = new Color(0.043f, 0.059f, 0.082f);
            PositionRect(wristOptBox.GetComponent<RectTransform>(), new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(18f, -300f), new Vector2(190f, 150f));

            var wristOptRaw = CreateRawImage("Wrist_Optical_Img", wristOptBox.transform, Color.white);
            PositionRect(wristOptRaw.rectTransform, new Vector2(0f, 0f), new Vector2(1f, 1f), new Vector2(0.5f, 0.5f), Vector2.zero, Vector2.zero);

            var wristOptBadge = CreateText("Wrist_Opt_Badge", wristOptBox.transform, "WRIST OPTICAL", font, 10, FontStyle.Bold, new Color(1f, 1f, 1f, 0.85f));
            PositionRect(wristOptBadge.rectTransform, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, 1f), new Vector2(8f, -6f), new Vector2(-16f, 16f));

            // Wrist Events Viewport Container
            var wristEvtBox = CreateUIElement("Wrist_Events_Box", sensoryPanelGo.transform);
            var wristEvtBoxImg = wristEvtBox.AddComponent<Image>();
            wristEvtBoxImg.color = new Color(0.043f, 0.059f, 0.082f);
            PositionRect(wristEvtBox.GetComponent<RectTransform>(), new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(222f, -300f), new Vector2(190f, 150f));

            var wristEvtRaw = CreateRawImage("Wrist_Events_Img", wristEvtBox.transform, Color.white);
            PositionRect(wristEvtRaw.rectTransform, new Vector2(0f, 0f), new Vector2(1f, 1f), new Vector2(0.5f, 0.5f), Vector2.zero, Vector2.zero);

            var wristEvtBadge = CreateText("Wrist_Evt_Badge", wristEvtBox.transform, "WRIST EVENTS (+/-)", font, 10, FontStyle.Bold, new Color(0f, 0.85f, 1f, 0.95f));
            PositionRect(wristEvtBadge.rectTransform, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, 1f), new Vector2(8f, -6f), new Vector2(-16f, 16f));

            var wristStatusText = CreateText("Wrist_Status", sensoryPanelGo.transform, "act: 0.0084  ·  ACTIVE", font, 12, FontStyle.Bold, new Color(0.22f, 0.83f, 0.33f));
            PositionRect(wristStatusText.rectTransform, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, 1f), new Vector2(18f, -456f), new Vector2(-36f, 20f));

            // Legend / Footer
            var legendText = CreateText("Sensory_Legend", sensoryPanelGo.transform,
                "<color=#00F0FF>■</color> ON Spike   <color=#FF3B5C>■</color> OFF Spike   ·   θ = 0.08   τ = 0.001",
                font, 11, FontStyle.Normal, new Color(0.55f, 0.58f, 0.62f));
            PositionRect(legendText.rectTransform, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, 1f), new Vector2(18f, -488f), new Vector2(-36f, 20f));

            // 9. Wiring SensoryStreamDisplay
            var sensory = canvasGo.AddComponent<SensoryStreamDisplay>();
            var sensorySo = new SerializedObject(sensory);
            sensorySo.FindProperty("playback").objectReferenceValue = playbackManager;
            sensorySo.FindProperty("globalOpticalImage").objectReferenceValue = globalOptRaw;
            sensorySo.FindProperty("globalEventsImage").objectReferenceValue = globalEvtRaw;
            sensorySo.FindProperty("globalStatusText").objectReferenceValue = globalStatusText;
            sensorySo.FindProperty("wristOpticalImage").objectReferenceValue = wristOptRaw;
            sensorySo.FindProperty("wristEventsImage").objectReferenceValue = wristEvtRaw;
            sensorySo.FindProperty("wristStatusText").objectReferenceValue = wristStatusText;
            sensorySo.FindProperty("headerTitleText").objectReferenceValue = sensoryTitleText;
            sensorySo.FindProperty("headerSubtitleText").objectReferenceValue = sensorySubText;
            sensorySo.FindProperty("legendText").objectReferenceValue = legendText;
            sensorySo.ApplyModifiedPropertiesWithoutUndo();

            // 10. Salvataggio scena
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
            go.layer = 5; // UI layer
            go.transform.SetParent(parent, false);
            return go;
        }

        private static RawImage CreateRawImage(string name, Transform parent, Color color)
        {
            var go = CreateUIElement(name, parent);
            var raw = go.AddComponent<RawImage>();
            raw.color = color;
            raw.raycastTarget = false;
            return raw;
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
            bool compareModels = false;

            string[] args = System.Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "-framesDir" && i < args.Length - 1) framesDir = args[i + 1];
                else if (args[i] == "-trajectoryJson" && i < args.Length - 1) trajJsonPath = args[i + 1];
                else if (args[i] == "-width" && i < args.Length - 1 && int.TryParse(args[i + 1], out int w)) width = w;
                else if (args[i] == "-height" && i < args.Length - 1 && int.TryParse(args[i + 1], out int h)) height = h;
                else if (args[i] == "-compareModels") compareModels = true;
            }

            if (!System.IO.File.Exists(trajJsonPath))
            {
                Debug.LogError($"[SPWM.HeadlessRenderVideo] Trajectory file not found: {trajJsonPath}");
                return;
            }

            System.IO.Directory.CreateDirectory(framesDir);

            // Load visualization scene (or rebuild to ensure latest room/camera/HUD setup)
            BuildScene();

            var playback = Object.FindFirstObjectByType<GhostPlaybackManager>();
            if (playback == null)
            {
                Debug.LogError("[SPWM.HeadlessRenderVideo] GhostPlaybackManager not found in scene!");
                return;
            }

            // Apply the flag before LoadTrajectory so the second rig is created
            playback.ForceCompareModelsMode = compareModels;
            playback.LoadTrajectory(trajJsonPath);

            var camObj = GameObject.FindWithTag("MainCamera");
            if (camObj == null) camObj = GameObject.Find("Main Camera");
            var cam = camObj != null ? camObj.GetComponent<Camera>() : Camera.main;
            if (cam == null)
            {
                Debug.LogError("[SPWM.HeadlessRenderVideo] Camera not found!");
                return;
            }

            // Configure Canvas at near-clip plane (0.06m) so UI always renders on top of 3D geometry without arm occlusion
            var canvas = Object.FindFirstObjectByType<Canvas>();
            if (canvas != null)
            {
                canvas.renderMode = RenderMode.ScreenSpaceCamera;
                canvas.worldCamera = cam;
                canvas.planeDistance = 0.06f;
                var scaler = canvas.GetComponent<CanvasScaler>();
                if (scaler != null)
                {
                    scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
                    scaler.referenceResolution = new Vector2(width >= height ? 1920 : 1080, width >= height ? 1080 : 1920);
                    scaler.matchWidthOrHeight = 0.5f;
                }
            }

            var sensory = Object.FindFirstObjectByType<SensoryStreamDisplay>();
            if (sensory != null)
            {
                sensory.EnsureCameras();
                sensory.OnPlaybackLoaded(playback);
            }

            // Set up RenderTexture for direct camera grab
            RenderTexture rt = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32);
            rt.antiAliasing = 4;
            cam.targetTexture = rt;
            Texture2D tex = new Texture2D(width, height, TextureFormat.RGB24, false);

            int totalFrames = playback.FrameCount;
            var tracker = camObj != null ? camObj.GetComponent<CinematicCameraTracker>() : null;
            var telemetry = Object.FindFirstObjectByType<TelemetryDisplay>();
            if (telemetry != null)
            {
                telemetry.Bind(playback);
            }

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

                if (sensory != null)
                {
                    sensory.RefreshSensors(force: true);
                }

                if (telemetry != null && refreshMethod != null)
                {
                    refreshMethod.Invoke(telemetry, new object[] { true });
                }

                Canvas.ForceUpdateCanvases();

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

            // 1. Anthropomorphic Eye-Level Head Sensor Camera with NeuromorphicEventBridge
            var camGo = new GameObject("NeuromorphicCamera");
            camGo.transform.SetParent(rootGo.transform, false);
            SPWM.Robotics.ArticulatedRobotRig.GetEyeLevelCameraPose(Vector3.zero, out Vector3 eyeCamPos, out Quaternion eyeCamRot);
            camGo.transform.localPosition = eyeCamPos;
            camGo.transform.localRotation = eyeCamRot;
            var cam = camGo.AddComponent<Camera>();
            cam.fieldOfView = 60f;
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
