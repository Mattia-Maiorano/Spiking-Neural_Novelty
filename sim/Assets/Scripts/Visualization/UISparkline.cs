using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace SPWM.Visualization
{
    /// <summary>
    /// Zero-dependency uGUI sparkline. Generates its own mesh in <see cref="OnPopulateMesh"/>:
    ///   1. faint "preview" polyline of the full series (what's coming),
    ///   2. gradient area fill under the revealed portion,
    ///   3. bright mitred polyline up to the playhead (fractional, interpolated),
    ///   4. glowing head marker,
    ///   5. optional dashed reference line (e.g. y = 1.0 for relative degradation baseline).
    /// One draw call, no textures required (uses the default UI white texture).
    /// </summary>
    [RequireComponent(typeof(CanvasRenderer))]
    public sealed class UISparkline : MaskableGraphic
    {
        [Header("Style")]
        [SerializeField, Min(0.5f)] private float lineThickness = 5f;
        [SerializeField] private Color lineColor = new Color(0.25f, 1f, 0.75f, 1f);
        [SerializeField] private Color previewColor = new Color(1f, 1f, 1f, 0.12f);
        [SerializeField] private Color fillTopColor = new Color(0.25f, 1f, 0.75f, 0.35f);
        [SerializeField] private Color fillBottomColor = new Color(0.25f, 1f, 0.75f, 0f);
        [SerializeField] private Color headColor = Color.white;
        [SerializeField, Min(0f)] private float headRadius = 9f;
        [SerializeField] private Color referenceColor = new Color(1f, 1f, 1f, 0.25f);

        [Header("Scale")]
        [SerializeField] private bool autoScale = true;
        [SerializeField] private float yMin = 0f;
        [SerializeField] private float yMax = 1.1f;
        [SerializeField] private bool includeZero = true;
        [SerializeField] private bool showReferenceLine = true;
        [SerializeField] private float referenceValue = 1f;
        [SerializeField, Range(0f, 0.3f)] private float verticalPadding = 0.08f;

        private readonly List<float> _values = new List<float>();
        private readonly List<float> _values2 = new List<float>(); // optional overlay series (model B)
        private Color _color2 = new Color(1f, 0.45f, 0f, 1f);
        private float _playhead; // continuous index into _values
        private float _lo, _hi;

        public float CurrentMin => _lo;
        public float CurrentMax => _hi;

        public override Texture mainTexture => s_WhiteTexture;

        /// <summary>Replace the series. Call once on load; then drive the reveal with <see cref="SetPlayhead"/>.</summary>
        public void SetSeries(IList<float> values)
        {
            _values.Clear();
            if (values != null) _values.AddRange(values);
            RecomputeRange();
            _playhead = 0f;
            SetVerticesDirty();
        }

        /// <summary>Overlay a second series (same x-axis, shared y-scale). Pass null to clear.</summary>
        public void SetSecondarySeries(IList<float> values, Color color)
        {
            _values2.Clear();
            if (values != null) _values2.AddRange(values);
            _color2 = color;
            RecomputeRange();
            SetVerticesDirty();
        }

        /// <summary>Recolor the primary series (line, fill, halo).</summary>
        public void SetPrimaryColor(Color c)
        {
            lineColor = c;
            fillTopColor = new Color(c.r, c.g, c.b, 0.25f);
            fillBottomColor = new Color(c.r, c.g, c.b, 0f);
            SetVerticesDirty();
        }

        /// <summary>Continuous index (e.g. GhostPlaybackManager.FramePosition). Fractional part interpolates the head.</summary>
        public void SetPlayhead(float index)
        {
            float clamped = Mathf.Clamp(index, 0f, Mathf.Max(0, _values.Count - 1));
            if (Mathf.Approximately(clamped, _playhead)) return;
            _playhead = clamped;
            SetVerticesDirty();
        }

        private void RecomputeRange()
        {
            if (!autoScale || _values.Count == 0) { _lo = yMin; _hi = yMax; return; }
            _lo = float.MaxValue; _hi = float.MinValue;
            foreach (float v in _values) { _lo = Mathf.Min(_lo, v); _hi = Mathf.Max(_hi, v); }
            foreach (float v in _values2) { _lo = Mathf.Min(_lo, v); _hi = Mathf.Max(_hi, v); }
            if (includeZero) _lo = Mathf.Min(_lo, 0f);
            if (showReferenceLine) { _lo = Mathf.Min(_lo, referenceValue); _hi = Mathf.Max(_hi, referenceValue); }
            if (_hi - _lo < 1e-5f) _hi = _lo + 1f;
            float pad = (_hi - _lo) * verticalPadding;
            _lo -= includeZero && _lo >= 0f ? 0f : pad;
            _hi += pad;
        }

        private Vector2 ToLocal(Rect r, float index, float value)
        {
            float x = _values.Count > 1 ? index / (_values.Count - 1) : 0f;
            float y = Mathf.InverseLerp(_lo, _hi, value);
            return new Vector2(r.xMin + x * r.width, r.yMin + y * r.height);
        }

        private static float SampleAt(IList<float> vals, float index)
        {
            int i0 = Mathf.FloorToInt(index);
            int i1 = Mathf.Min(i0 + 1, vals.Count - 1);
            return Mathf.Lerp(vals[i0], vals[i1], index - i0);
        }

        private List<Vector2> Revealed(Rect r, IList<float> vals)
        {
            int whole = Mathf.Min(Mathf.FloorToInt(_playhead), vals.Count - 1);
            var pts = new List<Vector2>(whole + 2);
            for (int i = 0; i <= whole; i++) pts.Add(ToLocal(r, i, vals[i]));
            if (_playhead - whole > 1e-4f && whole < vals.Count - 1) pts.Add(ToLocal(r, _playhead, SampleAt(vals, _playhead)));
            return pts;
        }

        protected override void OnPopulateMesh(VertexHelper vh)
        {
            vh.Clear();
            if (_values.Count < 2) return;
            Rect r = GetPixelAdjustedRect();

            // Full-series preview.
            var preview = new List<Vector2>(_values.Count);
            for (int i = 0; i < _values.Count; i++) preview.Add(ToLocal(r, i, _values[i]));

            // Revealed portion up to fractional playhead.
            var revealed = Revealed(r, _values);

            if (showReferenceLine && referenceValue >= _lo && referenceValue <= _hi)
                AddDashedLine(vh, r, ToLocal(r, 0, referenceValue).y, referenceColor, 2f, 10f, 8f);

            AddPolyline(vh, preview, lineThickness * 0.6f, previewColor);
            AddAreaFill(vh, revealed, r.yMin, fillTopColor, fillBottomColor);
            AddPolyline(vh, revealed, lineThickness, lineColor);

            if (_values2.Count == _values.Count)
            {
                var prev2 = new List<Vector2>(_values2.Count);
                for (int i = 0; i < _values2.Count; i++) prev2.Add(ToLocal(r, i, _values2[i]));
                var rev2 = Revealed(r, _values2);
                AddPolyline(vh, prev2, lineThickness * 0.6f, previewColor);
                AddAreaFill(vh, rev2, r.yMin, new Color(_color2.r, _color2.g, _color2.b, 0.25f), new Color(_color2.r, _color2.g, _color2.b, 0f));
                AddPolyline(vh, rev2, lineThickness, _color2);
                Vector2 head2 = rev2[rev2.Count - 1];
                AddDisc(vh, head2, headRadius * 2.2f, new Color(_color2.r, _color2.g, _color2.b, 0.25f));
                AddDisc(vh, head2, headRadius, _color2);
            }

            Vector2 head = revealed[revealed.Count - 1];
            AddDisc(vh, head, headRadius * 2.2f, new Color(lineColor.r, lineColor.g, lineColor.b, 0.25f)); // halo
            AddDisc(vh, head, headRadius, headColor);
        }

        // ---- mesh primitives ------------------------------------------------------------------

        private static void AddPolyline(VertexHelper vh, List<Vector2> pts, float thickness, Color c)
        {
            if (pts.Count < 2) return;
            float half = thickness * 0.5f;
            int start = vh.currentVertCount;
            for (int i = 0; i < pts.Count; i++)
            {
                Vector2 dirPrev = i > 0 ? (pts[i] - pts[i - 1]).normalized : (pts[1] - pts[0]).normalized;
                Vector2 dirNext = i < pts.Count - 1 ? (pts[i + 1] - pts[i]).normalized : dirPrev;
                Vector2 tangent = (dirPrev + dirNext).normalized;
                if (tangent.sqrMagnitude < 1e-6f) tangent = dirNext;
                Vector2 miter = new Vector2(-tangent.y, tangent.x);
                Vector2 normal = new Vector2(-dirNext.y, dirNext.x);
                float len = half / Mathf.Max(0.35f, Vector2.Dot(miter, normal)); // clamp spikes at sharp turns
                AddVert(vh, pts[i] + miter * len, c);
                AddVert(vh, pts[i] - miter * len, c);
            }
            for (int i = 0; i < pts.Count - 1; i++)
            {
                int a = start + i * 2;
                vh.AddTriangle(a, a + 1, a + 3);
                vh.AddTriangle(a, a + 3, a + 2);
            }
        }

        private static void AddAreaFill(VertexHelper vh, List<Vector2> pts, float baseline, Color top, Color bottom)
        {
            if (pts.Count < 2) return;
            int start = vh.currentVertCount;
            foreach (Vector2 p in pts)
            {
                AddVert(vh, p, top);
                AddVert(vh, new Vector2(p.x, baseline), bottom);
            }
            for (int i = 0; i < pts.Count - 1; i++)
            {
                int a = start + i * 2;
                vh.AddTriangle(a, a + 2, a + 3);
                vh.AddTriangle(a, a + 3, a + 1);
            }
        }

        private static void AddDisc(VertexHelper vh, Vector2 center, float radius, Color c, int segments = 20)
        {
            if (radius <= 0f) return;
            int centerIdx = vh.currentVertCount;
            AddVert(vh, center, c);
            for (int i = 0; i <= segments; i++)
            {
                float a = i / (float)segments * Mathf.PI * 2f;
                AddVert(vh, center + new Vector2(Mathf.Cos(a), Mathf.Sin(a)) * radius, c);
            }
            for (int i = 1; i <= segments; i++) vh.AddTriangle(centerIdx, centerIdx + i, centerIdx + i + 1);
        }

        private static void AddDashedLine(VertexHelper vh, Rect r, float y, Color c, float thickness, float dash, float gap)
        {
            float half = thickness * 0.5f;
            for (float x = r.xMin; x < r.xMax; x += dash + gap)
            {
                float x1 = Mathf.Min(x + dash, r.xMax);
                int s = vh.currentVertCount;
                AddVert(vh, new Vector2(x, y - half), c);
                AddVert(vh, new Vector2(x, y + half), c);
                AddVert(vh, new Vector2(x1, y + half), c);
                AddVert(vh, new Vector2(x1, y - half), c);
                vh.AddTriangle(s, s + 1, s + 2);
                vh.AddTriangle(s, s + 2, s + 3);
            }
        }

        private static void AddVert(VertexHelper vh, Vector2 p, Color c)
        {
            UIVertex v = UIVertex.simpleVert;
            v.position = p;
            v.color = c;
            v.uv0 = Vector4.zero;
            vh.AddVert(v);
        }

#if UNITY_EDITOR
        protected override void OnValidate()
        {
            base.OnValidate();
            RecomputeRange();
            SetVerticesDirty();
        }
#endif
    }
}
