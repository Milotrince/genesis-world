#version 330 core
///////////////////////////////////////////////////////////////////////////////
// Structs
///////////////////////////////////////////////////////////////////////////////

struct Material {
    vec3 emissive_factor;

#ifdef USE_METALLIC_MATERIAL
    vec4 base_color_factor;
    float metallic_factor;
    float roughness_factor;
#endif

#ifdef USE_GLOSSY_MATERIAL
    vec4 diffuse_factor;
    vec3 specular_factor;
    float glossiness_factor;
#endif

#ifdef HAS_NORMAL_TEX
    sampler2D normal_texture;
#endif
#ifdef HAS_OCCLUSION_TEX
    sampler2D occlusion_texture;
#endif
#ifdef HAS_EMISSIVE_TEX
#ifdef EMISSIVE_TEX_ARRAY
    sampler2DArray emissive_texture;
    float emissive_layer;
#else
    sampler2D emissive_texture;
#endif
#endif
#ifdef HAS_BASE_COLOR_TEX
#ifdef BASE_COLOR_TEX_ARRAY
    sampler2DArray base_color_texture;
    float base_color_layer;
#else
    sampler2D base_color_texture;
#endif
#endif
#ifdef HAS_METALLIC_ROUGHNESS_TEX
    sampler2D metallic_roughness_texture;
#endif
#ifdef HAS_DIFFUSE_TEX
    sampler2D diffuse_texture;
#endif
#ifdef HAS_SPECULAR_GLOSSINESS_TEX
    sampler2D specular_glossiness;
#endif
};

///////////////////////////////////////////////////////////////////////////////
// Uniforms
///////////////////////////////////////////////////////////////////////////////
uniform Material material;
uniform vec3 cam_pos;

#ifdef USE_IBL
uniform samplerCube diffuse_env;
uniform samplerCube specular_env;
#endif

///////////////////////////////////////////////////////////////////////////////
// Inputs
///////////////////////////////////////////////////////////////////////////////

in vec3 frag_position;
#ifdef NORMAL_LOC
in vec3 frag_normal;
#endif
#ifdef HAS_NORMAL_TEX
#ifdef TANGENT_LOC
#ifdef NORMAL_LOC
in mat3 tbn;
#endif
#endif
#endif
#ifdef TEXCOORD_0_LOC
in vec2 uv_0;
#endif
#ifdef TEXCOORD_1_LOC
in vec2 uv_1;
#endif
#ifdef COLOR_0_LOC
in vec4 color_multiplier;
#endif

///////////////////////////////////////////////////////////////////////////////
// OUTPUTS
///////////////////////////////////////////////////////////////////////////////

out vec4 frag_color;

///////////////////////////////////////////////////////////////////////////////
// Constants
///////////////////////////////////////////////////////////////////////////////
const float PI = 3.141592653589793;
const float min_roughness = 0.04;

///////////////////////////////////////////////////////////////////////////////
// Utility Functions
///////////////////////////////////////////////////////////////////////////////
vec4 srgb_to_linear(vec4 srgb)
{
#ifndef SRGB_CORRECTED
    // Fast Approximation
    //vec3 linOut = pow(srgbIn.xyz,vec3(2.2));
    //
    vec3 b_less = step(vec3(0.04045),srgb.xyz);
    vec3 lin_out = mix( srgb.xyz/vec3(12.92), pow((srgb.xyz+vec3(0.055))/vec3(1.055),vec3(2.4)), b_less );
    return vec4(lin_out, srgb.w);
#else
    return srgb;
#endif
}

#ifdef BASE_COLOR_TEX_LINEAR
vec4 linear_to_srgb(vec4 linear)
{
#ifndef SRGB_CORRECTED
    vec3 b_less = step(vec3(0.0031308), linear.xyz);
    vec3 srgb_out = mix(linear.xyz * vec3(12.92), vec3(1.055) * pow(linear.xyz, vec3(1.0 / 2.4)) - vec3(0.055), b_less);
    return vec4(srgb_out, linear.w);
#else
    return linear;
#endif
}
#endif

///////////////////////////////////////////////////////////////////////////////
// MAIN
///////////////////////////////////////////////////////////////////////////////
void main()
{

    // Compute albedo
    vec4 base_color = material.base_color_factor;
#ifdef HAS_BASE_COLOR_TEX
#ifdef BASE_COLOR_TEX_ARRAY
    vec4 base_color_texel = texture(material.base_color_texture, vec3(uv_0, material.base_color_layer));
#ifdef BASE_COLOR_TEX_LINEAR
    // A texture array holding linear values is encoded to sRGB, the encoding of every other base color
    base_color_texel = linear_to_srgb(base_color_texel);
#endif
    base_color = base_color * base_color_texel;
#else
    base_color = base_color * texture(material.base_color_texture, uv_0);
#endif
#endif

#ifdef COLOR_0_LOC
    base_color *= color_multiplier;
#endif

    frag_color = clamp(base_color, 0.0, 1.0);
}
