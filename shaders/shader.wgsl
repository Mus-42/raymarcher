struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2<f32>
};

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(in_vertex_index & 1u) * 2 - 1);
    let y = f32(i32(in_vertex_index >> 1u & 1u) * 2 - 1);
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

struct PointLight {
    pos: vec3<f32>,
    color: vec3<f32>,
}

const DIST_EPS = 0.001;
const DIST_MAX = 50.;
const DIST_REFLECTED_MAX = 10.;

const FOG_ABSORPTION = 0.04;
const FOG_REFLECTED_ABSORPTION = 0.1;
const FOG_COLOR = vec3<f32>(0.3, 0.3, 0.6);
const SKY_COLOR = FOG_COLOR;
const SUN_DIRECTION = vec3<f32>(0.3, 1., -0.1) / 1.1;
const SUN_COLOR = vec3<f32>(6., 6., 5.);

const MAX_RAYMARCH_ITERATIONS = 400;
const MAX_LIGHT_RAYMARCH_ITERATIONS = 20;
const MAX_REFLECTED_RAYMARCH_ITERATIONS = 30;

const MATERIAL_WHITE = 0u;
const MATERIAL_DIRT = 1u;
const MATERIAL_STONE = 2u;
const MATERIAL_GRASS = 3u;
const MATERIAL_IRON = 4u;
const MATERIAL_CHESSBOARD = 5u;
const MATERIAL_MIRROR = 6u;

const NUM_POINT_LIGHTS = 3;

var<private> POINT_LIGHTS: array<PointLight, NUM_POINT_LIGHTS> = array(
    PointLight(vec3<f32>(10., 3., 1.), vec3<f32>(3., 1., 1.)),
    PointLight(vec3<f32>(-10., 5., 15.), vec3<f32>(1., 3., 1.)),
    PointLight(vec3<f32>(-1., 0., -5.), vec3<f32>(1., 1., 3.)),
);

struct CameraTransform {
    view_proj: mat4x4<f32>,
};


@group(0) @binding(0)
var<uniform> camera_transform: CameraTransform;

@group(0) @binding(1)
var<uniform> screen_size: vec2<f32>;

@group(0) @binding(2)

var t_texmaps: texture_2d<f32>;
@group(0) @binding(3)
var s_texmaps: sampler;


fn horizontal_plane_sdf(p: vec3<f32>, y: f32) -> f32 {
    return abs(p.y - y);
}

fn sphere_sdf(p: vec3<f32>, r: f32) -> f32 {
    return length(p) - r;
}

fn cube_sdf(p: vec3<f32>, side: f32) -> f32 {
    let q = abs(p) - side;
    return length(max(q, vec3<f32>(0.))) + min(max(q.x, max(q.y, q.z)), 0.);
}

fn smax(a: f32, b: f32, k: f32) -> f32 {
    let h = max(k-abs(a-b),0.0);
    return max(a, b) + h*h*0.25/k;
}

fn hash(p: vec3<f32>) -> f32 {
    let v = 17.*fract(p*0.3183099+vec3(.11,.17,.13));
    return fract(v.x*v.y*v.z*(v.x+v.y+v.z));
}

// Stealed from https://iquilezles.org/articles/mandelbulb/
fn mandelbulb_sdf(p: vec3<f32>) -> f32 {
    var w = p;
    var m = dot(w,w);

    var trap = vec4(abs(w),m);
    var dz = 1.0;

	for(var i = 0; i < 4; i++) {
		dz = 8.0*pow(m,3.5)*dz + 1.0;

        let r = length(w);
        let b = 8.0*acos( w.y/r);
        let a = 8.0*atan2( w.x, w.z );
        w = p + pow(r,8.0) * vec3( sin(b)*sin(a), cos(b), sin(b)*cos(a) );

        trap = min( trap, vec4(abs(w),m) );

        m = dot(w,w);
		if(m > 256.0) {
            break;
        }
    }

    return 0.25*log(m)*sqrt(m)/dz;
}

// https://iquilezles.org/articles/distancefractals
fn mandelbrot_2d_sdf(c: vec2<f32>) -> f32 {
    let c2 = dot(c, c);
    if 256.0*c2*c2 - 96.0*c2 + 32.0*c.x - 3.0 < 0.0 {
        return 0.0;
    }
    if 16.0*(c2+2.0*c.x+1.0) - 1.0 < 0.0 {
        return 0.0;
    }

    var di = 1.0;
    var z = vec2(0.0);
    var m2 = 0.0;
    var dz = vec2(0.0);
    for(var i=0; i < 20; i++) {
        if(m2 > 1024.0) {
            di = 0.0;
            break;
        }
        dz = 2.0*vec2(z.x*dz.x-z.y*dz.y, z.x*dz.y + z.y*dz.x) + vec2(1.0,0.0);
        z = vec2( z.x*z.x - z.y*z.y, 2.0*z.x*z.y ) + c;
        m2 = dot(z,z);
    }

	var d = 0.5*sqrt(dot(z,z)/dot(dz,dz))*log(dot(z,z));
    if(di > 0.5) {
        d = 0.0;
    }
    return d;
}

struct SceneSDFOutput {
    dist: f32,
    material: u32,
};

fn combine(sdf_out: SceneSDFOutput, dist: f32, material: u32) -> SceneSDFOutput {
    if dist < sdf_out.dist {
        return SceneSDFOutput(dist, material);
    } else {
        return sdf_out;
    }
}

fn scene_sdf(p: vec3<f32>) -> SceneSDFOutput {
    var out = SceneSDFOutput(DIST_MAX, MATERIAL_WHITE);

    out = combine(out, horizontal_plane_sdf(p, -10.), MATERIAL_CHESSBOARD);
    out = combine(out, cube_sdf(p - vec3<f32>(-2., -1., 10.), 1.), MATERIAL_DIRT);
    out = combine(out, cube_sdf(p - vec3<f32>(10., -4., 1.), 1.), MATERIAL_WHITE);
    out = combine(
        out,
        max(
            cube_sdf(mat3x3f(0.35, 0.35, -0.87, -0.70, 0.78, 0., 0.61, 0.61, 0.5) * (1. / determinant(mat3x3f(0.35, 0.35, -0.87, -0.70, 0.78, 0., 0.61, 0.61, 0.5))) * (p - vec3<f32>(-2., -1., 14.)), 0.5),
            -sphere_sdf(p-vec3<f32>(-2., -1., 14.5), 0.7)
        ),
        MATERIAL_MIRROR
    );
    out = combine(out, sphere_sdf(p - vec3<f32>(2., -1., 10.), 1.), MATERIAL_MIRROR);
    out = combine(out, sphere_sdf(p - vec3<f32>(-2., -1., 17.) , 0.75), MATERIAL_STONE);
    out = combine(out, sphere_sdf(p - vec3<f32>(2., -1., 14.), 1.), MATERIAL_STONE);

    //out = combine(out, mandelbulb_sdf(p - vec3<f32>(0., 0., -10.)), MATERIAL_WHITE);
    //out = combine(out, max(mandelbrot_2d_sdf(p.xy - vec2<f32>(0.3, 0.)), cube_sdf(p - vec3<f32>(0., 0., -20.), 1.6)), MATERIAL_STONE);

    let q = p.x / 0.3;
    let pos = fract(q) * 0.3;
    let num = i32(floor(q));
    var materials = array(MATERIAL_STONE, MATERIAL_GRASS, MATERIAL_IRON, MATERIAL_DIRT, MATERIAL_CHESSBOARD, MATERIAL_MIRROR);
    out = combine(out, sphere_sdf(vec3<f32>(pos - 0.15, p.y + 1., p.z - 3.), 0.15), materials[(num % 6 + 6) % 6]);

    return out;
}

fn point_light_bloom(p: vec3<f32>, rd: vec3<f32>, max_t: f32) -> vec4<f32> {
    var out = vec4<f32>(0.);

    for (var i = 0; i < NUM_POINT_LIGHTS; i++) {
        let dir = POINT_LIGHTS[i].pos - p;
        let t = clamp(dot(dir, rd), 0., max_t);
        let closest = p + t * rd;
        let delta = POINT_LIGHTS[i].pos - closest;
        let dist = dot(delta, delta);
        let factor = pow(max(1. - dist, 0.), 10.);
        out += vec4<f32>(POINT_LIGHTS[i].color, 1.) * factor;
    }

    return out;
}

struct MaterialDesc {
    local_normal: vec3<f32>,
    diffuse_color: vec3<f32>,
    metallicity: f32,
}

fn dims_sum(v: vec3<f32>) -> f32 {
    return v.x + v.y + v.z;
}

fn uv_corner_dir(p: vec3<f32>, c: vec3<f32>) -> vec4<f32> {
    let i = floor(p);
    let f = fract(p);
    return vec4<f32>(hash(i + c), hash(i + c*7.), hash(i + c*3.), hash(i + c + 10.));
}

fn point_to_uv(p: vec3<f32>) -> vec2<f32> {
    return fract(p.xy * 0.3 + p.zy * 0.6 + p.xz * 0.4);
}


fn sample_material(p: vec3<f32>, material: u32) -> MaterialDesc {
    var metallicity = 0.;
    switch material {
        case MATERIAL_WHITE {
            return MaterialDesc(vec3<f32>(0., 0., 1.), vec3<f32>(1.), 0.);
        }
        case MATERIAL_CHESSBOARD {
            var ip = vec2<i32>(floor(p.xz));
            let col = vec3<f32>(f32((ip.x ^ ip.y) & 1)) * 0.5 + 0.3;
            return MaterialDesc(vec3<f32>(0., 0., 1.), col, 0.2);
        }
        case MATERIAL_MIRROR {
            return MaterialDesc(vec3<f32>(0., 0., 1.), SKY_COLOR, 1.);
        }
        case MATERIAL_GRASS {
            metallicity = 0.7;
        }
        case MATERIAL_IRON {
            metallicity = 0.9;
        }
        default {}
    }
    let uv = point_to_uv(p);
    let real_uv = uv * vec2<f32>(0.25, 0.5) + vec2<f32>(f32(material-1) * 0.25, 0.);
    let diff = textureSample(t_texmaps, s_texmaps, real_uv).rbg;
    let d = max(diff-vec3<f32>(0.7), vec3<f32>(0.));
    metallicity *= 0.1 + dot(d, d);
    //let normal = textureSample(t_texmaps, s_texmaps, real_uv + vec2<f32>(0., 0.5)).rbg;
    return MaterialDesc(vec3<f32>(0., 0., 1.), diff, metallicity);
}

fn get_normal(p: vec3<f32>, rd: vec3<f32>) -> vec3<f32> {
    let EPS = vec3<f32>(DIST_EPS, 0., 0.);

    let v1 = vec3<f32>(
        scene_sdf(p + EPS.xyy).dist,
        scene_sdf(p + EPS.yxy).dist,
        scene_sdf(p + EPS.yyx).dist,
    );

    let v2 = vec3<f32>(
        scene_sdf(p - EPS.xyy).dist,
        scene_sdf(p - EPS.yxy).dist,
        scene_sdf(p - EPS.yyx).dist,
    );

    let n = normalize(v1 - v2);

    return n;
}

fn raymarch_light(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
    var d = 0.;
    var md = 1.;
    for (var i = 0; i < MAX_LIGHT_RAYMARCH_ITERATIONS; i++) {
        let p = ro + rd * d;
        let dist = scene_sdf(p).dist;
        md = min(md, dist);
        d += dist;
        if d > DIST_MAX || dist < DIST_EPS {
            break;
        }
    }
    return md;
}

fn get_light(n: vec3<f32>, p: vec3<f32>, rd: vec3<f32>) -> vec3<f32> {
    // Directed light
    var d = raymarch_light(p, SUN_DIRECTION) * SUN_COLOR * max(dot(n, SUN_DIRECTION), 0.);
    for (var i = 0; i < NUM_POINT_LIGHTS; i++) {
        let r = POINT_LIGHTS[i].pos - p;
        let ld = normalize(r);
        let f = max(dot(n, ld), 0.) / length(r) * 30.;
        d += raymarch_light(p, ld) * f * POINT_LIGHTS[i].color;
    }
    return d;
}

fn get_diffuse_light(n: vec3<f32>, p: vec3<f32>, rd: vec3<f32>) -> vec3<f32> {
    // Diffuse light
    let r = reflect(rd, n);
    var q = vec3<f32>(0.);
    for (var i = 0; i < 10; i++) {
        let dir = normalize(r + vec3<f32>(hash(p + 0.1 + f32(i)), hash(p + 0.3 + f32(i)), hash(p + 0.4 + f32(i))) * 2. - 1.);
        q += raymarch_reflection2(p, dir) * max(dot(n, dir), 0.) ;
    }
    q /= 4.;
    return q;
}

fn raymarch_reflection2(ro: vec3<f32>, rd: vec3<f32>) -> vec3<f32> {
    var p = ro;
    var d = 0.;
    var col = SKY_COLOR;
    for (var i = 0; i < MAX_REFLECTED_RAYMARCH_ITERATIONS; i++) {
        let sdf_out = scene_sdf(p);
        if sdf_out.dist > DIST_REFLECTED_MAX {
            break;
        }
        if sdf_out.dist < DIST_EPS {
            let n = get_normal(p, rd);
            let material = sample_material(p - n * sdf_out.dist, sdf_out.material);
            let light = get_light(n, p + n * 0.1, rd);
            col = material.diffuse_color * light;
            break;
        }
        d += sdf_out.dist;
        p += rd * sdf_out.dist;
    }
    let fog_factor = exp(-d * FOG_REFLECTED_ABSORPTION);
    let bloom = point_light_bloom(ro, rd, d+0.1);
    return mix(mix(FOG_COLOR, col, fog_factor), bloom.rgb, bloom.w * (fog_factor * 0.5 + 0.5));
}

fn raymarch_reflection(ro: vec3<f32>, rd: vec3<f32>) -> vec3<f32> {
    var p = ro;
    var d = 0.;
    var col = SKY_COLOR;
    for (var i = 0; i < MAX_REFLECTED_RAYMARCH_ITERATIONS; i++) {
        let sdf_out = scene_sdf(p);
        if sdf_out.dist > DIST_REFLECTED_MAX {
            break;
        }
        if sdf_out.dist < DIST_EPS {
            let n = get_normal(p, rd);
            let material = sample_material(p - n * sdf_out.dist, sdf_out.material);
            let light = get_light(n, p + n * 0.1, rd);
            col = material.diffuse_color;
            col *= light;
            if material.metallicity > 0. {
                let reflected = raymarch_reflection2(p + n * 0.1, reflect(rd, n));
                col = mix(col, reflected * (light * 0.3 + 0.7), material.metallicity);
            }
            break;
        }
        d += sdf_out.dist;
        p += rd * sdf_out.dist;
    }
    let fog_factor = exp(-d * FOG_REFLECTED_ABSORPTION);
    let bloom = point_light_bloom(ro, rd, d + 0.1);
    return mix(mix(FOG_COLOR, col, fog_factor), bloom.rgb, bloom.w * (fog_factor * 0.5 + 0.5));
}

fn raymarch(ro: vec3<f32>, rd: vec3<f32>) -> vec3<f32> {
    var p = ro;
    var d = 0.;
    var col = SKY_COLOR;
    for (var i = 0; i < MAX_RAYMARCH_ITERATIONS; i++) {
        let sdf_out = scene_sdf(p);
        if sdf_out.dist > DIST_MAX {
            break;
        }
        if sdf_out.dist < DIST_EPS {
            let n = get_normal(p, rd);
            let material = sample_material(p - n * sdf_out.dist, sdf_out.material);
            let light = get_light(n, p + n * 0.1, rd);
            col = material.diffuse_color;
            col *= light;
            if material.metallicity > 0. {
                let reflected = raymarch_reflection(p + n * 0.1, reflect(rd, n));
                col = mix(col, reflected * (light * 0.3 + 0.7), material.metallicity);
            }
            break;
        }
        d += sdf_out.dist;
        p += rd * sdf_out.dist;
    }
    let fog_factor = exp(-d * FOG_ABSORPTION);
    let bloom = point_light_bloom(ro, rd, d+0.1);
    return mix(mix(FOG_COLOR, col, fog_factor), bloom.rgb, bloom.w * (fog_factor * 0.5 + 0.5));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ro_raw = camera_transform.view_proj * vec4<f32>(0., 0., 0., 1.);
    let ro = ro_raw.xyz / ro_raw.w;
    let scale_factor = screen_size / min(screen_size.x, screen_size.y) * 0.5;
    let rd_raw = camera_transform.view_proj * vec4<f32>(1., in.uv.yx * scale_factor.yx, 1.);
    let rd = normalize(rd_raw.xyz / rd_raw.w - ro);
    let col = raymarch(ro, rd);
    return vec4<f32>(col, 1.);
}
