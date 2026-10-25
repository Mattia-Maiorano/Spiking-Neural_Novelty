// SPWM-v0.4.0 Holographic Ghost (Built-in Render Pipeline)
// Unlit, transparent, additive neon emission with fresnel rim, scrolling scanlines and flicker.
// A depth pre-pass hides the ghost's own back-facing/overlapping limbs for a clean silhouette.
Shader "SPWM/HolographicGhost"
{
    Properties
    {
        [HDR] _Color        ("Core Emission", Color) = (0.0, 0.85, 1.0, 1.0)
        [HDR] _RimColor     ("Rim Emission", Color) = (0.15, 1.0, 0.65, 1.0)
        _Alpha              ("Core Opacity", Range(0, 1)) = 0.22
        _RimPower           ("Rim Power", Range(0.5, 8)) = 2.2
        _RimIntensity       ("Rim Intensity", Range(0, 4)) = 1.6
        _ScanDensity        ("Scanline Density (lines / m)", Range(0, 400)) = 120
        _ScanSpeed          ("Scanline Speed", Range(-5, 5)) = 0.6
        _ScanStrength       ("Scanline Strength", Range(0, 1)) = 0.35
        _FlickerStrength    ("Flicker Strength", Range(0, 0.5)) = 0.06
        _Inflate            ("Normal Inflate (m)", Range(0, 0.02)) = 0.004
    }

    SubShader
    {
        Tags { "Queue"="Transparent" "RenderType"="Transparent" "IgnoreProjector"="True" "PreviewType"="Sphere" }
        LOD 100

        CGINCLUDE
        #include "UnityCG.cginc"
        float _Inflate;

        float4 InflatedClipPos(float4 vertex, float3 normal)
        {
            vertex.xyz += normalize(normal) * _Inflate;
            return UnityObjectToClipPos(vertex);
        }
        ENDCG

        // Pass 1: depth-only so overlapping ghost limbs don't double-blend.
        Pass
        {
            Name "DepthPrepass"
            ZWrite On
            ColorMask 0
            Cull Back

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            struct appdata { float4 vertex : POSITION; float3 normal : NORMAL; };
            float4 vert(appdata v) : SV_POSITION { return InflatedClipPos(v.vertex, v.normal); }
            fixed4 frag() : SV_Target { return 0; }
            ENDCG
        }

        // Pass 2: additive holographic emission.
        Pass
        {
            Name "Hologram"
            ZWrite Off
            ZTest LEqual
            Cull Back
            Blend SrcAlpha One

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_fog

            fixed4 _Color, _RimColor;
            float _Alpha, _RimPower, _RimIntensity;
            float _ScanDensity, _ScanSpeed, _ScanStrength, _FlickerStrength;

            struct appdata { float4 vertex : POSITION; float3 normal : NORMAL; };
            struct v2f
            {
                float4 pos      : SV_POSITION;
                float3 worldPos : TEXCOORD0;
                float3 worldN   : TEXCOORD1;
                UNITY_FOG_COORDS(2)
            };

            v2f vert(appdata v)
            {
                v2f o;
                o.pos = InflatedClipPos(v.vertex, v.normal);
                o.worldPos = mul(unity_ObjectToWorld, v.vertex).xyz;
                o.worldN = UnityObjectToWorldNormal(v.normal);
                UNITY_TRANSFER_FOG(o, o.pos);
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                float3 n = normalize(i.worldN);
                float3 viewDir = normalize(_WorldSpaceCameraPos - i.worldPos);
                float fresnel = pow(1.0 - saturate(dot(n, viewDir)), _RimPower);

                float scan = 0.5 + 0.5 * sin((i.worldPos.y * _ScanDensity + _Time.y * _ScanSpeed * 10.0) * 3.14159);
                float scanMask = lerp(1.0, scan, _ScanStrength);
                float flicker = 1.0 - _FlickerStrength * (0.5 + 0.5 * sin(_Time.y * 37.0) * sin(_Time.y * 11.0));

                float3 rgb = _Color.rgb * scanMask + _RimColor.rgb * fresnel * _RimIntensity;
                float a = saturate((_Alpha * scanMask + fresnel * _RimIntensity * 0.6) * flicker);

                fixed4 col = fixed4(rgb * flicker, a);
                UNITY_APPLY_FOG_COLOR(i.fogCoord, col, fixed4(0, 0, 0, 0)); // additive: fog toward black
                return col;
            }
            ENDCG
        }
    }
    FallBack Off
}
