//! Bodies out of a parsed transmit file, as kernel-neutral B-reps in millimeters.

use std::collections::HashMap;

use shapecraft_kernel::brep::{BodyKind, Brep, Curve, Edge, Face, Fin, Nurbs, NurbsSurface, Surface};
use shapecraft_kernel::{DMat3, DVec2, DVec3};

use crate::blend::{self, Support};
use crate::read::{Node, Val, XtFile};
use crate::{Error, Result};

/// Parasolid lengths are meters.
pub const MM: f64 = 1000.0;

/// One body of the file with what the file says about it.
#[derive(Clone, Debug)]
pub struct Body {
    /// Index of the BODY node.
    pub node: u32,
    pub brep: Brep,
    pub name: Option<String>,
    pub color: Option<[f32; 3]>,
    /// Every attribute on the body: definition name → values (strings, numbers).
    pub attributes: Vec<Attribute>,
    /// Each face's node id, in the order of `brep.faces`: what the file's own records call the face by.
    pub face_ids: Vec<i64>,
    /// The triangles of faces that lie on a MESH surface (a facet body: an imported STL Shapr3D kept as a mesh).
    pub mesh: Option<Mesh>,
}

/// Triangles in millimeters; corners index `points`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mesh {
    pub points: Vec<DVec3>,
    pub triangles: Vec<[u32; 3]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Attribute {
    pub definition: String,
    pub values: Vec<AttrValue>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AttrValue {
    Ints(Vec<i64>),
    Reals(Vec<f64>),
    Text(String),
    Vectors(Vec<[f64; 3]>),
    Other,
}

/// Every BODY node in the file, in file order.
pub fn bodies(f: &XtFile) -> Result<Vec<Body>> {
    let body_ty = 12;
    f.nodes.iter().filter(|n| n.ty == body_ty).map(|n| body(f, n)).collect()
}

struct Cx<'a> {
    f: &'a XtFile,
    b: Brep,
    surfaces: HashMap<u32, (usize, bool)>,
    curves: HashMap<u32, (usize, bool)>,
    vertices: HashMap<u32, usize>,
    edges: HashMap<u32, usize>,
}

fn bad(msg: impl Into<String>) -> Error {
    Error::Bad(msg.into())
}

/// Which way the file's cone v runs against the side of the cone we build (see `uv_map`). Not yet confirmed on
/// data: the reference's formula and what we measured of cone axes disagree.
const CONE_V_SIGN: f64 = 1.0;

/// (u, v) in a file's surface parametrization → (u, v) in ours: optionally swapped, then scaled.
#[derive(Clone, Copy, Debug, PartialEq)]
struct UvMap {
    swap: bool,
    scale: DVec2,
}

impl UvMap {
    const IDENTITY: UvMap = UvMap { swap: false, scale: DVec2::ONE };
    fn apply(&self, p: DVec2) -> DVec2 {
        let p = if self.swap { DVec2::new(p.y, p.x) } else { p };
        p * self.scale
    }
}

impl<'a> Cx<'a> {
    fn node(&self, p: u32) -> Option<&'a Node> {
        if p == 0 { None } else { self.f.node(p) }
    }
    fn get(&self, n: &'a Node, field: &str) -> Option<&'a Val> {
        self.f.field(n, field)
    }
    fn ptr(&self, n: &Node, field: &str) -> u32 {
        self.f.field(n, field).map(Val::ptr).unwrap_or(0)
    }
    fn vec(&self, n: &Node, field: &str) -> Result<DVec3> {
        let v = self
            .f
            .field(n, field)
            .and_then(Val::vec3)
            .ok_or_else(|| bad(format!("{} #{} has no {field}", self.f.type_name(n.ty), n.index)))?;
        Ok(DVec3::from_array(v))
    }
    fn real(&self, n: &Node, field: &str) -> Result<f64> {
        self.f.field(n, field).and_then(Val::real).ok_or_else(|| bad(format!("{} #{} has no {field}", self.f.type_name(n.ty), n.index)))
    }
    fn minus(&self, n: &Node, field: &str) -> bool {
        self.f.field(n, field).and_then(Val::char) == Some(b'-')
    }

    /// Surface index in the B-rep and whether its natural normal points opposite the file's.
    fn surface(&mut self, p: u32) -> Result<(usize, bool)> {
        if let Some(&s) = self.surfaces.get(&p) {
            return Ok(s);
        }
        let n = self.node(p).ok_or_else(|| bad(format!("missing surface #{p}")))?;
        let name = self.f.type_name(n.ty).to_string();
        let (s, flip) = match name.as_str() {
            "PLANE" => {
                (Surface::Plane { origin: self.vec(n, "pvec")? * MM, normal: self.vec(n, "normal")?, x: self.vec(n, "x_axis")? }, false)
            }
            "CYLINDER" => (
                Surface::Cylinder {
                    origin: self.vec(n, "pvec")? * MM,
                    axis: self.vec(n, "axis")?,
                    x: self.vec(n, "x_axis")?,
                    radius: self.real(n, "radius")? * MM,
                },
                false,
            ),
            "CONE" => {
                // Measured on a Shapr3D chamfer (m1): the radius grows ALONG the axis and the natural normal points
                // out, like OCCT's cone. (The reference's formula reads "P - vA"; the data says otherwise.)
                let a = self.real(n, "sin_half_angle")?.atan2(self.real(n, "cos_half_angle")?);
                (
                    Surface::Cone {
                        origin: self.vec(n, "pvec")? * MM,
                        axis: self.vec(n, "axis")?,
                        x: self.vec(n, "x_axis")?,
                        radius: self.real(n, "radius")? * MM,
                        half_angle: a,
                    },
                    false,
                )
            }
            "SPHERE" => (
                Surface::Sphere {
                    center: self.vec(n, "centre")? * MM,
                    axis: self.vec(n, "axis")?,
                    x: self.vec(n, "x_axis")?,
                    radius: self.real(n, "radius")? * MM,
                },
                false,
            ),
            "TORUS" => (
                Surface::Torus {
                    center: self.vec(n, "centre")? * MM,
                    axis: self.vec(n, "axis")?,
                    x: self.vec(n, "x_axis")?,
                    major: self.real(n, "major_radius")? * MM,
                    minor: self.real(n, "minor_radius")? * MM,
                },
                false,
            ),
            "B_SURFACE" => (Surface::BSpline(self.nurbs_surface(self.ptr(n, "nurbs"))?), false),
            "OFFSET_SURF" => {
                let (basis, bflip) = self.surface(self.ptr(n, "surface"))?;
                let basis_node = self.node(self.ptr(n, "surface"));
                let sense = if basis_node.is_some_and(|b| self.minus(b, "sense")) { -1.0 } else { 1.0 };
                let along = if bflip { -1.0 } else { 1.0 };
                (Surface::Offset { basis, distance: self.real(n, "offset")? * MM * sense * along }, bflip)
            }
            "SWEPT_SURF" => {
                let (curve, _) = self.curve(self.ptr(n, "section"))?;
                (Surface::Extrusion { curve, dir: self.vec(n, "sweep")? }, false)
            }
            "SPUN_SURF" => {
                // u runs along the profile and v around the axis; OCCT swaps them, so the normal flips.
                let (curve, _) = self.curve(self.ptr(n, "profile"))?;
                (Surface::Revolution { curve, origin: self.vec(n, "base")? * MM, axis: self.vec(n, "axis")? }, true)
            }
            "BLENDED_EDGE" if self.get(n, "blend_type").and_then(Val::char) == Some(b'R') => {
                // u follows the spine's own parameter and v runs from the first support to the second (the other
                // way when the spine's sense is negative), so the natural normal flips with the spine.
                let supports = self.blend_supports(n)?;
                let (spine, reversed) = self.curve(self.ptr(n, "spine"))?;
                let centers = match &self.b.curves[spine] {
                    Curve::Through(pts) => pts.clone(),
                    Curve::Circle { center, normal, x, radius } => {
                        let (x, y) = (x.normalize_or_zero(), normal.normalize_or_zero().cross(x.normalize_or_zero()));
                        (0..=48)
                            .map(|k| {
                                let (s, c) = (std::f64::consts::TAU * k as f64 / 48.0).sin_cos();
                                *center + *radius * (c * x + s * y)
                            })
                            .collect()
                    }
                    Curve::Ellipse { center, normal, x, major, minor } => {
                        let (x, y) = (x.normalize_or_zero(), normal.normalize_or_zero().cross(x.normalize_or_zero()));
                        (0..=64)
                            .map(|k| {
                                let (s, c) = (std::f64::consts::TAU * k as f64 / 64.0).sin_cos();
                                *center + *major * c * x + *minor * s * y
                            })
                            .collect()
                    }
                    _ => return Err(bad(format!("blend #{p} has a spine we can't follow yet"))),
                };
                let s = blend::rolling_ball(&self.b.surfaces, supports, &centers).map_err(|e| bad(format!("blend #{p}: {e}")))?;
                (Surface::BSpline(s), reversed)
            }
            other => return Err(bad(format!("surface type {other} (#{p}) is not supported yet"))),
        };
        let i = self.b.surfaces.len();
        self.b.surfaces.push(s);
        self.surfaces.insert(p, (i, flip));
        Ok((i, flip))
    }

    /// Distinct knots and their multiplicities. `count` is the NURBS node's own count (`n_knots`): the arrays
    /// can be longer than that, padded with unset knots and zero multiplicities (seen in real Shapr3D files).
    fn knots(&self, mult: u32, knots: u32, count: Option<usize>) -> Result<(Vec<f64>, Vec<u32>)> {
        let mut ks = match self.node(knots).and_then(|n| n.fields.first()) {
            Some(Val::Arr(a)) => a.iter().map(Val::real).collect::<Vec<_>>(),
            _ => return Err(bad("b-spline without knots")),
        };
        let mut ms = match self.node(mult).and_then(|n| n.fields.first()) {
            Some(Val::Arr(a)) => a.iter().map(Val::int).collect::<Vec<_>>(),
            _ => return Err(bad("b-spline without multiplicities")),
        };
        // Without a usable count, the padding is the run of unset knots / zero multiplicities at the end.
        let n = count.filter(|&n| n > 0 && n <= ks.len() && n <= ms.len()).unwrap_or_else(|| {
            let real = ks.iter().take_while(|k| k.is_some()).count();
            let used = ms.iter().take_while(|m| m.is_some_and(|m| m > 0)).count();
            real.min(used)
        });
        ks.truncate(n);
        ms.truncate(n);
        let ks = ks.into_iter().map(|k| k.ok_or_else(|| bad("b-spline knot unset"))).collect::<Result<Vec<_>>>()?;
        let ms = ms
            .into_iter()
            .map(|m| m.filter(|m| (1..=64).contains(m)).map(|m| m as u32).ok_or_else(|| bad("b-spline knot multiplicity out of range")))
            .collect::<Result<Vec<_>>>()?;
        if ks.len() < 2 {
            return Err(bad("b-spline with fewer than two knots"));
        }
        if ks.windows(2).any(|w| w[1] < w[0]) {
            return Err(bad("b-spline knots out of order"));
        }
        Ok((ks, ms))
    }

    /// Control vertices as (cartesian point, weight) pairs; `dim` 3/4 in space, 2/3 in a surface's (u, v).
    fn vertices(&self, p: u32, dim: usize, rational: bool, count: usize) -> Result<Vec<(Vec<f64>, f64)>> {
        let mut raw = match self.node(p).and_then(|n| n.fields.first()) {
            Some(Val::Arr(a)) => a.iter().map(|v| v.real().unwrap_or(0.0)).collect::<Vec<_>>(),
            _ => return Err(bad("b-spline without vertices")),
        };
        // Like the knots, the array may run past the node's own count.
        if count > 0 && count.saturating_mul(dim) <= raw.len() {
            raw.truncate(count * dim);
        }
        if dim == 0 || !raw.len().is_multiple_of(dim) {
            return Err(bad("b-spline vertex count"));
        }
        Ok(raw
            .chunks(dim)
            .map(|c| {
                if rational {
                    let w = c[dim - 1];
                    let w = if w.abs() < 1e-300 { 1.0 } else { w };
                    (c[..dim - 1].iter().map(|x| x / w).collect(), w)
                } else {
                    (c.to_vec(), 1.0)
                }
            })
            .collect())
    }

    fn nurbs_surface(&self, p: u32) -> Result<NurbsSurface> {
        let n = self.node(p).ok_or_else(|| bad("b-surface without NURBS data"))?;
        let int = |f: &str| self.get(n, f).and_then(Val::int).unwrap_or(0).max(0) as usize;
        let flag = |f: &str| self.get(n, f).and_then(Val::bool).unwrap_or(false);
        let rational = flag("rational");
        let (nu, nv, dim) = (int("n_u_vertices"), int("n_v_vertices"), int("vertex_dim"));
        let verts = self.vertices(self.ptr(n, "bspline_vertices"), dim, rational, nu * nv)?;
        if verts.len() != nu * nv {
            return Err(bad(format!("b-surface has {} vertices, expected {nu}×{nv}", verts.len())));
        }
        let count = |f: &str| self.get(n, f).and_then(Val::int).and_then(|c| usize::try_from(c).ok());
        let (u_knots, u_mults) = self.knots(self.ptr(n, "u_knot_mult"), self.ptr(n, "u_knots"), count("n_u_knots"))?;
        let (v_knots, v_mults) = self.knots(self.ptr(n, "v_knot_mult"), self.ptr(n, "v_knots"), count("n_v_knots"))?;
        let poles = verts.iter().map(|(c, _)| DVec3::new(c[0], c[1], c.get(2).copied().unwrap_or(0.0)) * MM).collect();
        let weights = rational.then(|| verts.iter().map(|(_, w)| *w).collect());
        Ok(NurbsSurface {
            u_degree: int("u_degree") as u32,
            v_degree: int("v_degree") as u32,
            nu,
            nv,
            poles,
            weights,
            u_knots,
            u_mults,
            v_knots,
            v_mults,
            // The full knot vectors are stored, so the open form gives the same surface.
            u_periodic: false,
            v_periodic: false,
        })
    }

    fn nurbs_curve(&self, p: u32) -> Result<(Nurbs<Vec<f64>>, usize)> {
        let n = self.node(p).ok_or_else(|| bad("b-curve without NURBS data"))?;
        let int = |f: &str| self.get(n, f).and_then(Val::int).unwrap_or(0).max(0) as usize;
        let rational = self.get(n, "rational").and_then(Val::bool).unwrap_or(false);
        let dim = int("vertex_dim");
        let verts = self.vertices(self.ptr(n, "bspline_vertices"), dim, rational, int("n_vertices"))?;
        let count = self.get(n, "n_knots").and_then(Val::int).and_then(|c| usize::try_from(c).ok());
        let (knots, mults) = self.knots(self.ptr(n, "knot_mult"), self.ptr(n, "knots"), count)?;
        let weights = rational.then(|| verts.iter().map(|(_, w)| *w).collect());
        let poles = verts.into_iter().map(|(c, _)| c).collect();
        Ok((Nurbs { degree: int("degree") as u32, poles, weights, knots, mults, periodic: false }, if rational { dim - 1 } else { dim }))
    }

    /// Curve index and whether the file's curve runs against its natural direction.
    fn curve(&mut self, p: u32) -> Result<(usize, bool)> {
        if let Some(&c) = self.curves.get(&p) {
            return Ok(c);
        }
        let n = self.node(p).ok_or_else(|| bad(format!("missing curve #{p}")))?;
        let name = self.f.type_name(n.ty).to_string();
        let reversed = self.minus(n, "sense");
        let c = match name.as_str() {
            "LINE" => Curve::Line { origin: self.vec(n, "pvec")? * MM, dir: self.vec(n, "direction")? },
            "CIRCLE" => Curve::Circle {
                center: self.vec(n, "centre")? * MM,
                normal: self.vec(n, "normal")?,
                x: self.vec(n, "x_axis")?,
                radius: self.real(n, "radius")? * MM,
            },
            "ELLIPSE" => Curve::Ellipse {
                center: self.vec(n, "centre")? * MM,
                normal: self.vec(n, "normal")?,
                x: self.vec(n, "x_axis")?,
                major: self.real(n, "major_radius")? * MM,
                minor: self.real(n, "minor_radius")? * MM,
            },
            "B_CURVE" => {
                let (nb, dim) = self.nurbs_curve(self.ptr(n, "nurbs"))?;
                if dim != 3 {
                    return Err(bad("b-curve in space must be 3D"));
                }
                Curve::BSpline(Nurbs {
                    degree: nb.degree,
                    poles: nb.poles.iter().map(|c| DVec3::new(c[0], c[1], c[2]) * MM).collect(),
                    weights: nb.weights,
                    knots: nb.knots,
                    mults: nb.mults,
                    periodic: false,
                })
            }
            "TRIMMED_CURVE" => {
                // The edge's vertices trim it; keep the basis curve (with this curve's sense on top).
                let (c, r) = self.curve(self.ptr(n, "basis_curve"))?;
                let out = (c, r != reversed);
                self.curves.insert(p, out);
                return Ok(out);
            }
            "SP_CURVE" => {
                let (surface, _) = self.surface(self.ptr(n, "surface"))?;
                let map = self.uv_map(self.ptr(n, "surface")).ok_or_else(|| bad(format!("SP-curve #{p} on a surface we can't map")))?;
                let (nb, dim) = self.nurbs_curve(self.node(self.ptr(n, "b_curve")).map(|b| self.ptr(b, "nurbs")).unwrap_or(0))?;
                if dim != 2 {
                    return Err(bad("SP-curve must be 2D"));
                }
                Curve::OnSurface {
                    surface,
                    uv: Nurbs {
                        degree: nb.degree,
                        // An affine map of the poles is the same map of the curve.
                        poles: nb.poles.iter().map(|c| map.apply(DVec2::new(c[0], c[1]))).collect(),
                        weights: nb.weights,
                        knots: nb.knots,
                        mults: nb.mults,
                        periodic: false,
                    },
                }
            }
            "INTERSECTION" => {
                let chart = self.node(self.ptr(n, "chart")).ok_or_else(|| bad("intersection without a chart"))?;
                let pts = match self.get(chart, "hvec") {
                    Some(Val::Arr(a)) => a
                        .iter()
                        .filter_map(|h| match h {
                            Val::Reals(r) if r.len() >= 3 => Some(DVec3::new(r[0], r[1], r[2]) * MM),
                            _ => None,
                        })
                        .collect::<Vec<_>>(),
                    _ => Vec::new(),
                };
                if pts.len() < 2 {
                    return Err(bad("intersection curve with fewer than two points"));
                }
                // The chart is coarse; put dense points on the exact curve where both surfaces have a closed form.
                Curve::Through(self.exact_intersection(n, &pts)?.unwrap_or(pts))
            }
            other => return Err(bad(format!("curve type {other} (#{p}) is not supported yet"))),
        };
        let i = self.b.curves.len();
        self.b.curves.push(c);
        self.curves.insert(p, (i, reversed));
        Ok((i, reversed))
    }

    /// A fin's own SP-curve (a tolerant edge's fins carry one each) when it lies in the face's surface.
    fn fin_pcurve(&mut self, c: u32, surface: usize) -> Option<usize> {
        if c == 0 {
            return None;
        }
        let (i, _) = self.curve(c).ok()?;
        matches!(self.b.curves.get(i), Some(Curve::OnSurface { surface: s, .. }) if *s == surface).then_some(i)
    }

    /// How hard projecting onto the surface under an SP-curve (through a trimmed curve) is, lowest first sorted
    /// last: planes 0, other elementary surfaces 1, anything else 2 (returned negated for an ascending sort).
    fn projection_cost(&self, c: u32) -> i32 {
        let mut n = self.node(c);
        for _ in 0..4 {
            match n {
                Some(x) if self.f.type_name(x.ty) == "TRIMMED_CURVE" => n = self.node(self.ptr(x, "basis_curve")),
                _ => break,
            }
        }
        let surface = n.filter(|x| self.f.type_name(x.ty) == "SP_CURVE").and_then(|x| self.node(self.ptr(x, "surface")));
        -match surface.map(|s| self.f.type_name(s.ty)) {
            Some("PLANE") => 0,
            Some("CYLINDER" | "CONE" | "SPHERE" | "TORUS") => 1,
            _ => 2,
        }
    }

    /// How a surface's (u, v) in the file maps onto the (u, v) of the surface we build from it; None where the
    /// two parametrizations aren't related simply (blends we approximate).
    fn uv_map(&self, p: u32) -> Option<UvMap> {
        let n = self.node(p)?;
        let curve_scale = |c: u32| -> Option<f64> {
            let c = self.node(c)?;
            match self.f.type_name(c.ty) {
                // A line's parameter is its length (meters here, millimeters there); the others are angles or
                // spline parameters, which don't change.
                "LINE" => Some(MM),
                "CIRCLE" | "B_CURVE" => Some(1.0),
                _ => None,
            }
        };
        Some(match self.f.type_name(n.ty) {
            "PLANE" => UvMap { swap: false, scale: DVec2::splat(MM) },
            "CYLINDER" => UvMap { swap: false, scale: DVec2::new(1.0, MM) },
            "SPHERE" | "TORUS" | "B_SURFACE" => UvMap::IDENTITY,
            "CONE" => {
                // The file's v runs along the axis; ours runs along the cone's side, from the same circle.
                let (s, c) = (self.real(n, "sin_half_angle").ok()?, self.real(n, "cos_half_angle").ok()?);
                if c.abs() < 1e-12 || !(s.is_finite() && c.is_finite()) {
                    return None;
                }
                UvMap { swap: false, scale: DVec2::new(1.0, CONE_V_SIGN * MM / c.abs()) }
            }
            "SWEPT_SURF" => UvMap { swap: false, scale: DVec2::new(curve_scale(self.ptr(n, "section"))?, MM) },
            // u along the profile, v around the axis; ours the other way round.
            "SPUN_SURF" => UvMap { swap: true, scale: DVec2::new(1.0, curve_scale(self.ptr(n, "profile"))?) },
            "OFFSET_SURF" => self.uv_map(self.ptr(n, "surface"))?,
            _ => return None,
        })
    }

    /// A blend's supports, each with the offset that carries it through the ball's center.
    fn blend_supports(&mut self, n: &Node) -> Result<[Support; 2]> {
        let ptrs = match self.get(n, "surface") {
            Some(Val::Arr(a)) if a.len() == 2 => [a[0].ptr(), a[1].ptr()],
            _ => return Err(bad("blend without two supports")),
        };
        let range = match self.get(n, "range") {
            Some(Val::Arr(a)) if a.len() == 2 => [a[0].real().unwrap_or(0.0), a[1].real().unwrap_or(0.0)],
            Some(Val::Reals(r)) if r.len() == 2 => [r[0], r[1]],
            _ => return Err(bad("blend without a range")),
        };
        let mut out = [(0, 0.0); 2];
        for k in 0..2 {
            let (s, flip) = self.surface(ptrs[k])?;
            // (Tried: offsetting along our turned normal builds these, but a blend rolled on our approximation of
            // another blend lands up to a millimeter off; refuse rather than build a bad solid.)
            if flip {
                return Err(bad("blend on a surface whose normal we turn"));
            }
            let sense = if self.node(ptrs[k]).is_some_and(|sn| self.minus(sn, "sense")) { -1.0 } else { 1.0 };
            out[k] = (s, range[k] * MM * sense);
        }
        Ok(out)
    }

    /// Dense points on an intersection curve, or None when one of its surfaces has no closed form here.
    fn exact_intersection(&mut self, n: &Node, chart: &[DVec3]) -> Result<Option<Vec<DVec3>>> {
        let ptrs = match self.get(n, "surface") {
            Some(Val::Arr(a)) if a.len() == 2 => [a[0].ptr(), a[1].ptr()],
            _ => return Ok(None),
        };
        let kind = |p: u32| self.node(p).map(|sn| self.f.type_name(sn.ty).to_string()).unwrap_or_default();
        let (ka, kb) = (kind(ptrs[0]), kind(ptrs[1]));
        // A blend's boundary: where the ball touches the support (the other surface is a construction surface).
        let bound = match (ka.as_str(), kb.as_str()) {
            ("BLEND_BOUND", "BLEND_BOUND") => return Ok(None),
            ("BLEND_BOUND", _) => Some((ptrs[0], ptrs[1])),
            (_, "BLEND_BOUND") => Some((ptrs[1], ptrs[0])),
            _ => None,
        };
        if let Some((bb, support)) = bound {
            let Some(blend) = self.node(bb).and_then(|b| self.node(self.ptr(b, "blend"))) else { return Ok(None) };
            if self.f.type_name(blend.ty) != "BLENDED_EDGE" || self.ptr(blend, "spine") == 0 {
                return Ok(None);
            }
            let Ok(s) = self.blend_supports(blend) else { return Ok(None) };
            let Ok((x, false)) = self.surface(support) else { return Ok(None) };
            let Some(&(_, off)) = s.iter().find(|(k, _)| *k == x) else { return Ok(None) };
            let surfaces = &self.b.surfaces;
            // The ball touching the support at p has its center straight out from p.
            return Ok(blend::densify(chart, 32, |p| {
                let (f, n) = blend::project(surfaces, x, p)?;
                let c = blend::on_both(surfaces, s[0], s[1], f + off * n)?;
                Some(blend::project(surfaces, x, c)?.0)
            }));
        }
        let (Ok((a, false)), Ok((b, false))) = (self.surface(ptrs[0]), self.surface(ptrs[1])) else { return Ok(None) };
        let surfaces = &self.b.surfaces;
        // Closest points on a spline surface are only local: on a thread's flank they can jump a turn. Keep the
        // chart there (blends above still use them, near their supports).
        if [a, b].iter().any(|&k| matches!(surfaces.get(k), Some(Surface::BSpline(_)))) {
            return Ok(None);
        }
        Ok(blend::densify(chart, 32, |p| blend::on_both(surfaces, (a, 0.0), (b, 0.0), p)))
    }

    fn vertex(&mut self, p: u32) -> Result<Option<usize>> {
        if p == 0 {
            return Ok(None);
        }
        if let Some(&v) = self.vertices.get(&p) {
            return Ok(Some(v));
        }
        let n = self.node(p).ok_or_else(|| bad(format!("missing vertex #{p}")))?;
        let pt = self.node(self.ptr(n, "point")).ok_or_else(|| bad("vertex without a point"))?;
        let i = self.b.vertices.len();
        self.b.vertices.push(self.vec(pt, "pvec")? * MM);
        self.vertices.insert(p, i);
        Ok(Some(i))
    }

    /// The fins around an edge (ring through `other`).
    fn edge_fins(&self, e: &Node) -> Vec<&'a Node> {
        let mut out = Vec::new();
        let start = self.ptr(e, "fin");
        let mut p = start;
        while let Some(fin) = self.node(p) {
            out.push(fin);
            p = self.ptr(fin, "other");
            if p == start || out.len() > 64 {
                break;
            }
        }
        out
    }

    fn edge(&mut self, p: u32, fin_curve: u32) -> Result<usize> {
        if let Some(&e) = self.edges.get(&p) {
            return Ok(e);
        }
        let n = self.node(p).ok_or_else(|| bad(format!("missing edge #{p}")))?;
        let fins = self.edge_fins(n);
        // The positive fin's forward vertex ends the edge, the negative fin's starts it.
        let end = fins.iter().find(|f| !self.minus(f, "sense")).map(|f| self.ptr(f, "vertex")).unwrap_or(0);
        let start = fins.iter().find(|f| self.minus(f, "sense")).map(|f| self.ptr(f, "vertex")).unwrap_or(0);
        let cp = self.ptr(n, "curve");
        let tolerance = self.get(n, "tolerance").and_then(Val::real).map(|t| t * MM);
        let (curve, reversed) = if cp != 0 {
            self.curve(cp)?
        } else {
            // A tolerant edge: its fins carry SP-curves. Take the first one we can place on its surface.
            let mut cands: Vec<u32> = fins.iter().map(|f| self.ptr(f, "curve")).filter(|&c| c != 0).collect();
            if fin_curve != 0 && !cands.contains(&fin_curve) {
                cands.push(fin_curve);
            }
            // The kernel projects the edge onto its other faces itself: cheap onto a plane or a cylinder, slow onto
            // a spline surface. So place it on the costliest surface first.
            cands.sort_by_key(|&c| self.projection_cost(c));
            let mut found = Err(bad(format!("edge #{p} has no curve")));
            for c in cands {
                found = self.curve(c);
                if found.is_ok() {
                    break;
                }
            }
            found?
        };
        let (start, end) = (self.vertex(start)?, self.vertex(end)?);
        // Points on an intersection end where the edge's vertices are, not a few microns off.
        if let (Some(Curve::Through(pts)), Some(s), Some(e)) = (self.b.curves.get_mut(curve), start, end) {
            let (s, e) = (self.b.vertices[s], self.b.vertices[e]);
            let (s, e) = if reversed { (e, s) } else { (s, e) };
            let n = pts.len();
            if n >= 2 && pts[0].distance(s) < 0.05 && pts[n - 1].distance(e) < 0.05 {
                pts[0] = s;
                pts[n - 1] = e;
            }
        }
        let i = self.b.edges.len();
        self.b.edges.push(Edge { curve: Some(curve), reversed, start, end, tolerance });
        self.edges.insert(p, i);
        Ok(i)
    }

    fn face(&mut self, n: &'a Node) -> Result<Face> {
        let (surface, natural_flip) = self.surface(self.ptr(n, "surface"))?;
        let surf_minus = self.node(self.ptr(n, "surface")).is_some_and(|s| self.minus(s, "sense"));
        let flip = (self.minus(n, "sense") != surf_minus) != natural_flip;
        let mut loops = Vec::new();
        let mut lp = self.ptr(n, "loop");
        let mut guard = 0;
        while let Some(l) = self.node(lp) {
            let mut fins = Vec::new();
            let first = self.ptr(l, "fin");
            let mut fp = first;
            while let Some(fin) = self.node(fp) {
                let ep = self.ptr(fin, "edge");
                if ep != 0 {
                    let edge = self.edge(ep, self.ptr(fin, "curve"))?;
                    let pcurve = self.fin_pcurve(self.ptr(fin, "curve"), surface);
                    fins.push(Fin { edge, forward: !self.minus(fin, "sense"), pcurve });
                }
                fp = self.ptr(fin, "forward");
                if fp == first || fins.len() > 1 << 20 {
                    break;
                }
            }
            if !fins.is_empty() {
                loops.push(fins);
            }
            lp = self.ptr(l, "next");
            guard += 1;
            if guard > 1 << 20 {
                return Err(bad("loop chain runaway"));
            }
        }
        let color = self.attributes(n).into_iter().find(|a| a.definition == "SDL/TYSA_COLOUR").and_then(|a| match a.values.first() {
            Some(AttrValue::Reals(r)) if r.len() >= 3 => Some([r[0] as f32, r[1] as f32, r[2] as f32]),
            _ => None,
        });
        Ok(Face { surface, flip, loops, color })
    }

    /// The attributes chained off a node's `attributes_groups`.
    fn attributes(&self, n: &Node) -> Vec<Attribute> {
        let mut out = Vec::new();
        let mut p = self.ptr(n, "attributes_groups");
        let mut guard = 0;
        while let Some(a) = self.node(p) {
            if self.f.type_name(a.ty) == "ATTRIBUTE" {
                let def = self.node(self.ptr(a, "definition"));
                let definition = def
                    .and_then(|d| self.node(self.ptr(d, "identifier")))
                    .and_then(|id| match id.fields.first() {
                        Some(Val::Bytes(b)) => Some(String::from_utf8_lossy(b).into_owned()),
                        _ => None,
                    })
                    .unwrap_or_default();
                let values = match self.get(a, "fields") {
                    Some(Val::Arr(ps)) => ps.iter().map(|v| self.values(v.ptr())).collect(),
                    _ => Vec::new(),
                };
                out.push(Attribute { definition, values });
            }
            p = self.ptr(a, "next");
            guard += 1;
            if guard > 4096 {
                break;
            }
        }
        out
    }

    fn values(&self, p: u32) -> AttrValue {
        let Some(n) = self.node(p) else { return AttrValue::Other };
        let first = n.fields.first();
        match (self.f.type_name(n.ty), first) {
            ("INT_VALUES", Some(Val::Arr(a))) => AttrValue::Ints(a.iter().filter_map(Val::int).collect()),
            ("REAL_VALUES", Some(Val::Arr(a))) => AttrValue::Reals(a.iter().filter_map(Val::real).collect()),
            ("CHAR_VALUES", Some(Val::Bytes(b))) => AttrValue::Text(String::from_utf8_lossy(b).into_owned()),
            ("UNICODE_VALUES", Some(Val::Arr(a))) => {
                let units: Vec<u16> = a.iter().filter_map(|v| v.int().map(|i| i as u16)).collect();
                AttrValue::Text(String::from_utf16_lossy(&units))
            }
            ("POINT_VALUES" | "VECTOR_VALUES" | "DIRECTION_VALUES" | "AXIS_VALUES", Some(Val::Arr(a))) => {
                AttrValue::Vectors(a.iter().filter_map(Val::vec3).collect())
            }
            _ => AttrValue::Other,
        }
    }
}

fn body(f: &XtFile, n: &Node) -> Result<Body> {
    let mut cx =
        Cx { f, b: Brep::default(), surfaces: HashMap::new(), curves: HashMap::new(), vertices: HashMap::new(), edges: HashMap::new() };
    cx.b.kind = match f.field(n, "body_type").and_then(Val::int) {
        Some(2) => BodyKind::Wire,
        Some(3) => BodyKind::Sheet,
        _ => BodyKind::Solid,
    };
    // Faces hang off the shells of the body's regions: back faces through `face`/`next`, front faces through
    // `front_face`/`next_front`. Group them by the shell they are a back face of.
    let mut shells: Vec<(u32, Vec<usize>)> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut rp = cx.ptr(n, "region");
    let mut guard = 0usize;
    let mut face_nodes = Vec::new();
    while let Some(region) = cx.node(rp) {
        let mut sp = cx.ptr(region, "shell");
        while let Some(shell) = cx.node(sp) {
            for (head, next) in [("face", "next"), ("front_face", "next_front")] {
                let mut fp = cx.ptr(shell, head);
                while let Some(face) = cx.node(fp) {
                    if seen.insert(fp) {
                        face_nodes.push(face);
                    }
                    fp = cx.ptr(face, next);
                    guard += 1;
                    if guard > 1 << 22 {
                        return Err(bad("face chain runaway"));
                    }
                }
            }
            // Wire bodies (including construction axes) carry their edges directly on the shell, with no faces.
            if cx.b.kind == BodyKind::Wire {
                let mut edge = cx.ptr(shell, "edge");
                let mut visited = std::collections::HashSet::new();
                while let Some(e) = cx.node(edge) {
                    if !visited.insert(edge) || visited.len() > 1 << 22 {
                        return Err(bad("wire edge chain runaway"));
                    }
                    cx.edge(edge, 0)?;
                    edge = cx.ptr(e, "next");
                }
            }
            sp = cx.ptr(shell, "next");
            guard += 1;
            if guard > 1 << 22 {
                return Err(bad("shell chain runaway"));
            }
        }
        rp = cx.ptr(region, "next");
        guard += 1;
        if guard > 1 << 22 {
            return Err(bad("region chain runaway"));
        }
    }
    let mut mesh: Option<Mesh> = None;
    let mut face_ids = Vec::new();
    for face in face_nodes {
        if let Some(surf) = cx.node(cx.ptr(face, "surface"))
            && f.type_name(surf.ty) == "MESH"
        {
            facets(&cx, surf, mesh.get_or_insert_with(Mesh::default))?;
            continue;
        }
        let fi = cx.b.faces.len();
        let built = cx.face(face)?;
        cx.b.faces.push(built);
        face_ids.push(match f.field(face, "node_id") {
            Some(Val::Int(id)) => *id,
            _ => -1,
        });
        let shell = cx.ptr(face, "shell");
        match shells.iter_mut().find(|(s, _)| *s == shell) {
            Some((_, v)) => v.push(fi),
            None => shells.push((shell, vec![fi])),
        }
    }
    cx.b.shells = shells.into_iter().map(|(_, v)| v).collect();
    let attributes = cx.attributes(n);
    let text = |def: &str| {
        attributes.iter().find(|a| a.definition == def).and_then(|a| match a.values.first() {
            Some(AttrValue::Text(t)) => Some(t.clone()),
            _ => None,
        })
    };
    let name = text("SDL/TYSA_UNAME").or_else(|| text("SDL/TYSA_NAME"));
    let color = attributes.iter().find(|a| a.definition == "SDL/TYSA_COLOUR").and_then(|a| match a.values.first() {
        Some(AttrValue::Reals(r)) if r.len() >= 3 => Some([r[0] as f32, r[1] as f32, r[2] as f32]),
        _ => None,
    });
    join_split_walls(&mut cx.b, &mut face_ids);
    Ok(Body { node: n.index, brep: cx.b, name, color, attributes, face_ids, mesh })
}

/// A clamped B-spline surface value at `(u, v)`, for comparing two surfaces' geometry.
fn bspline_at(s: &NurbsSurface, u: f64, v: f64) -> DVec3 {
    let expand = |k: &[f64], m: &[u32]| -> Vec<f64> { k.iter().zip(m).flat_map(|(k, m)| std::iter::repeat_n(*k, *m as usize)).collect() };
    let knots = expand(&s.u_knots, &s.u_mults);
    let vknots = expand(&s.v_knots, &s.v_mults);
    let nu = blend::basis(&knots, s.u_degree as usize, u);
    let nv = blend::basis(&vknots, s.v_degree as usize, v);
    let weight = |i: usize, j: usize| s.weights.as_ref().map_or(1.0, |w| w[i * s.nv + j]);
    let (mut p, mut w) = (DVec3::ZERO, 0.0);
    for (i, bu) in nu.iter().enumerate() {
        for (j, bv) in nv.iter().enumerate() {
            let k = bu * bv * weight(i, j);
            p += k * s.poles[i * s.nv + j];
            w += k;
        }
    }
    if w != 0.0 { p / w } else { p }
}

/// Whether two B-spline surfaces describe the same geometry on their shared interior, and the second is closed
/// along u (its two u ends meet).
fn surfaces_coincide(a: &NurbsSurface, b: &NurbsSurface) -> bool {
    let (Some((au0, au1, v0, v1)), Some((bu0, bu1, _, _))) = (domain(a), domain(b)) else {
        return false;
    };
    if a.u_degree != b.u_degree || a.v_degree != b.v_degree || a.nv != b.nv {
        return false;
    }
    if a.v_knots != b.v_knots || a.v_mults != b.v_mults {
        return false;
    }
    // The u knot vectors differ only at the ends of a clamped closed surface; the interior is the same. Sample
    // inside both u domains and the shared v domain, never at fixed numbers that may lie outside them.
    let (u0, u1) = (au0.max(bu0), au1.min(bu1));
    if u1 <= u0 {
        return false;
    }
    let at = |lo: f64, hi: f64, f: f64| lo + (hi - lo) * f;
    for &fu in &[0.25, 0.5, 0.75] {
        for &fv in &[0.25, 0.5, 0.75] {
            let (u, v) = (at(u0, u1, fu), at(v0, v1, fv));
            if bspline_at(a, u, v).distance(bspline_at(b, u, v)) > 1e-6 {
                return false;
            }
        }
    }
    (0..=4).all(|i| {
        let v = at(v0, v1, 0.25 + 0.5 * i as f64 / 4.0);
        bspline_at(b, bu0, v).distance(bspline_at(b, bu1, v)) < 1e-2
    })
}

/// The `(u0, u1, v0, v1)` parameter domain of a well-formed clamped B-spline surface, or `None` when its counts
/// don't add up (pole grid, weights, knots per degree) or a number isn't finite: file data is hostile, and
/// `bspline_at` indexes by those counts.
fn domain(s: &NurbsSurface) -> Option<(f64, f64, f64, f64)> {
    let (pu, pv) = (s.u_degree as usize, s.v_degree as usize);
    let expand = |k: &[f64], m: &[u32]| -> Vec<f64> { k.iter().zip(m).flat_map(|(k, m)| std::iter::repeat_n(*k, *m as usize)).collect() };
    let (ku, kv) = (expand(&s.u_knots, &s.u_mults), expand(&s.v_knots, &s.v_mults));
    let ok = pu >= 1
        && pv >= 1
        && s.nu > pu
        && s.nv > pv
        && s.u_knots.len() == s.u_mults.len()
        && s.v_knots.len() == s.v_mults.len()
        && ku.len() == s.nu + pu + 1
        && kv.len() == s.nv + pv + 1
        && s.poles.len() == s.nu * s.nv
        && s.weights.as_ref().is_none_or(|w| w.len() == s.poles.len())
        && ku.iter().chain(&kv).all(|k| k.is_finite())
        && s.poles.iter().all(|p| p.is_finite());
    if !ok {
        return None;
    }
    let (u0, u1, v0, v1) = (ku[pu], ku[ku.len() - 1 - pu], kv[pv], kv[kv.len() - 1 - pv]);
    (u1 > u0 && v1 > v0).then_some((u0, u1, v0, v1))
}

/// Where a fin starts and ends (vertex indices), following the edge's and the fin's senses.
fn fin_ends(b: &Brep, f: &Fin) -> (Option<usize>, Option<usize>) {
    let e = &b.edges[f.edge];
    let (s, t) = if e.reversed { (e.end, e.start) } else { (e.start, e.end) };
    if f.forward { (s, t) } else { (t, s) }
}

/// Whether the fins form one closed ring: each fin ends where the next begins.
fn loop_closes(b: &Brep, fins: &[Fin]) -> bool {
    if fins.len() < 3 || fins.iter().any(|f| matches!(fin_ends(b, f), (None, _) | (_, None))) {
        return false;
    }
    (0..fins.len()).all(|i| fin_ends(b, &fins[i]).1 == fin_ends(b, &fins[(i + 1) % fins.len()]).0)
}

/// Some files store a closed wall as two faces that share both of the wall's seam edges: one carries the top
/// rim, the other the bottom rim and the holes. Neither closes on its own (the first has no far rim to bound it),
/// so the shell comes out invalid. Join each such pair into the one face the geometry describes, as the STEP form
/// of the same body has it: both seam edges stay, one on each side of the parameter rectangle. `face_ids` (one per
/// face) loses the joined-away face's entry with it.
fn join_split_walls(b: &mut Brep, face_ids: &mut Vec<i64>) {
    loop {
        let mut found = None;
        'shells: for shell in &b.shells {
            for (ia, &a) in shell.iter().enumerate() {
                if b.faces[a].loops.len() != 1 {
                    continue;
                }
                for &c in &shell[ia + 1..] {
                    if let Some(pair) = split_wall_pair(b, a, c) {
                        found = Some(pair);
                        break 'shells;
                    }
                }
            }
        }
        let Some((keep, drop, seams)) = found else { break };
        let moved = b.faces[keep].loops[0].len();
        let mut outer = std::mem::take(&mut b.faces[keep].loops[0]);
        outer.extend(b.faces[drop].loops[0].iter().copied().filter(|f| !seams.contains(&f.edge)));
        b.faces[drop].loops[0] = outer;
        // The moved fins carried curves in the other face's (u, v); let the kernel project them instead.
        for fin in &mut b.faces[drop].loops[0][moved..] {
            fin.pcurve = None;
        }
        b.faces.remove(keep);
        if keep < face_ids.len() {
            face_ids.remove(keep);
        }
        for shell in &mut b.shells {
            shell.retain(|&f| f != keep);
            for f in shell.iter_mut() {
                if *f > keep {
                    *f -= 1;
                }
            }
        }
    }
}

/// The fins of `face`'s outer loop besides the wall's two seam edges.
fn outer_chain(b: &Brep, face: usize, seams: &[usize; 2]) -> Vec<Fin> {
    b.faces[face].loops[0].iter().copied().filter(|f| !seams.contains(&f.edge)).collect()
}

/// If `a` (one loop) and `c` are the two halves of one closed wall, how to join them: the face whose outer loop
/// to keep, the one whose rim joins it, and the two seam edges between them. `None` if no order closes the ring.
fn split_wall_pair(b: &Brep, a: usize, c: usize) -> Option<(usize, usize, [usize; 2])> {
    let (fa, fc) = (&b.faces[a], &b.faces[c]);
    if fc.loops.is_empty() || fa.loops[0].len() < 4 || fc.loops[0].len() < 4 {
        return None;
    }
    let (Surface::BSpline(sa), Surface::BSpline(sc)) = (&b.surfaces[fa.surface], &b.surfaces[fc.surface]) else {
        return None;
    };
    if !surfaces_coincide(sa, sc) {
        return None;
    }
    // Both loops must share exactly two edges, once each: the wall's two seam edges.
    let a_edges: Vec<usize> = fa.loops[0].iter().map(|f| f.edge).collect();
    let seams: Vec<usize> = a_edges.iter().copied().filter(|e| fc.loops[0].iter().any(|f| f.edge == *e)).collect();
    if seams.len() != 2 || seams[0] == seams[1] {
        return None;
    }
    let seams = [seams[0], seams[1]];
    let ca = outer_chain(b, a, &seams);
    let cc = outer_chain(b, c, &seams);
    if ca.is_empty() || cc.is_empty() {
        return None;
    }
    // Keep the face whose loop plus the other's rim closes; the one-loop rim strip usually comes first.
    let try_order = |keep: usize, drop: usize, chain: &[Fin]| {
        let mut fins = b.faces[keep].loops[0].clone();
        fins.extend_from_slice(chain);
        loop_closes(b, &fins).then_some((keep, drop, seams))
    };
    try_order(a, c, &cc).or_else(|| try_order(c, a, &ca))
}

/// Every number a node's array field holds, in order.
fn numbers(v: Option<&Val>) -> Vec<f64> {
    match v {
        Some(Val::Reals(r)) => r.clone(),
        Some(Val::Arr(a)) => a.iter().filter_map(Val::real).collect(),
        Some(Val::Vec3(v)) => v.to_vec(),
        _ => Vec::new(),
    }
}

/// A comb's values: its teeth (chunks) one after another, cut to its count.
fn comb(cx: &Cx<'_>, comb: &Node, count_field: &str) -> Vec<f64> {
    let mut out = Vec::new();
    if let Some(Val::Arr(teeth)) = cx.get(comb, "teeth") {
        for t in teeth.iter().map(Val::ptr).filter(|p| *p != 0) {
            if let Some(tooth) = cx.node(t) {
                out.extend(numbers(cx.get(tooth, "values")));
            }
        }
    }
    let n = cx.get(comb, count_field).and_then(Val::int).unwrap_or(i64::MAX).max(0) as usize;
    out.truncate(if count_field == "n_vectors" { n.saturating_mul(3) } else { n });
    out
}

/// Adds a MESH surface's triangles to `mesh`: positions (meters) through the mesh's transform, 1-based corners.
fn facets(cx: &Cx<'_>, surf: &Node, mesh: &mut Mesh) -> Result<()> {
    let psm = cx.node(cx.ptr(surf, "psm_imesh")).ok_or_else(|| bad(format!("mesh #{} has no facets", surf.index)))?;
    let pool = cx.node(cx.ptr(psm, "position_pool")).ok_or_else(|| bad("mesh without positions"))?;
    let idx = cx.node(cx.ptr(psm, "position_indices")).ok_or_else(|| bad("mesh without triangles"))?;
    let (rot, shift) = match cx.node(cx.ptr(surf, "transform")) {
        Some(t) => {
            let m = numbers(cx.get(t, "rotation_matrix"));
            let v = cx.get(t, "translation_vector").and_then(Val::vec3).unwrap_or([0.0; 3]);
            let rot = if m.len() == 9 {
                DMat3::from_cols_array(&[m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]]).transpose()
            } else {
                DMat3::IDENTITY
            };
            (rot, DVec3::from_array(v))
        }
        None => (DMat3::IDENTITY, DVec3::ZERO),
    };
    let base = mesh.points.len() as u32;
    let xyz = comb(cx, pool, "n_vectors");
    if xyz.len() > 30_000_000 {
        return Err(bad("mesh too large"));
    }
    mesh.points.extend(xyz.as_chunks::<3>().0.iter().map(|c| (rot * DVec3::new(c[0], c[1], c[2]) + shift) * MM));
    let count = (xyz.len() / 3) as u32;
    let corners = comb(cx, idx, "n_integers");
    for t in corners.as_chunks::<3>().0 {
        let k = [t[0], t[1], t[2]].map(|i| i as i64 - 1);
        if k.iter().all(|i| *i >= 0 && (*i as u32) < count) {
            mesh.triangles.push(k.map(|i| base + i as u32));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A closed-in-`u` strip: `u = 0` and `u = 1` land on the same poles.
    fn wall_surface() -> NurbsSurface {
        let poles =
            [(0.0, 0.0), (1.0, 0.0), (0.0, 0.0)].into_iter().flat_map(|(x, y)| [DVec3::new(x, y, 0.0), DVec3::new(x, y, 1.0)]).collect();
        NurbsSurface {
            u_degree: 1,
            v_degree: 1,
            nu: 3,
            nv: 2,
            poles,
            weights: None,
            u_knots: vec![0.0, 0.5, 1.0],
            u_mults: vec![2, 1, 2],
            v_knots: vec![0.0, 1.0],
            v_mults: vec![2, 2],
            u_periodic: false,
            v_periodic: false,
        }
    }

    fn fin(edge: usize) -> Fin {
        Fin { edge, forward: true, pcurve: None }
    }

    /// A wall stored as two faces sharing both seam edges: one loop with the top rim, one with the bottom rim and
    /// a hole. Joining them must leave one face with both rims and keep the hole.
    #[test]
    fn two_faces_of_one_closed_wall_become_one() {
        let mut b = Brep {
            // b(0) and t(1) are the seam ends; m1(2) is on the top rim, m2(3) on the bottom.
            vertices: vec![DVec3::ZERO, DVec3::Z, DVec3::new(1.0, 0.0, 1.0), DVec3::new(1.0, 0.0, 0.0)],
            edges: vec![
                Edge { curve: None, reversed: false, start: Some(0), end: Some(1), tolerance: None }, // seam A
                Edge { curve: None, reversed: false, start: Some(1), end: Some(2), tolerance: None }, // top rim
                Edge { curve: None, reversed: false, start: Some(2), end: Some(1), tolerance: None },
                Edge { curve: None, reversed: false, start: Some(1), end: Some(0), tolerance: None }, // seam B
                Edge { curve: None, reversed: false, start: Some(0), end: Some(3), tolerance: None }, // bottom rim
                Edge { curve: None, reversed: false, start: Some(3), end: Some(0), tolerance: None },
            ],
            surfaces: vec![Surface::BSpline(wall_surface()), Surface::BSpline(wall_surface())],
            faces: vec![
                // Top rim strip: seam A, the top rim, seam B.
                Face { surface: 0, flip: false, loops: vec![vec![fin(0), fin(1), fin(2), fin(3)]], color: None },
                // The rest: both seams, the bottom rim, and a hole.
                Face { surface: 1, flip: false, loops: vec![vec![fin(3), fin(0), fin(4), fin(5)], vec![fin(1), fin(2)]], color: None },
            ],
            shells: vec![vec![0, 1]],
            ..Brep::default()
        };
        let mut ids = vec![10, 11];
        join_split_walls(&mut b, &mut ids);
        assert_eq!(b.faces.len(), 1, "the two halves join into one face");
        assert_eq!(ids, [11], "the joined face keeps the id of the face that holds the holes");
        assert_eq!(b.faces[0].loops.len(), 2, "the hole survives");
        let outer: Vec<usize> = b.faces[0].loops[0].iter().map(|f| f.edge).collect();
        assert_eq!(outer, [0, 1, 2, 3, 4, 5], "both rims and both seams, each seam once");
        assert_eq!(b.shells, vec![vec![0]]);
    }

    #[test]
    fn coincident_surfaces_are_recognised_and_shifted_ones_are_not() {
        let a = wall_surface();
        let mut c = wall_surface();
        assert!(surfaces_coincide(&a, &c));
        for pole in &mut c.poles {
            pole.x += 1.0;
        }
        assert!(!surfaces_coincide(&a, &c));
    }

    /// File data is hostile: counts that don't add up must be refused, not indexed.
    #[test]
    fn malformed_surfaces_are_refused_without_panicking() {
        let a = wall_surface();
        let mut short = wall_surface();
        short.u_degree = 9;
        assert!(!surfaces_coincide(&a, &short));
        let mut few_poles = wall_surface();
        few_poles.poles.truncate(2);
        assert!(!surfaces_coincide(&a, &few_poles));
        let mut nan = wall_surface();
        nan.u_knots[1] = f64::NAN;
        assert!(!surfaces_coincide(&nan, &a));
    }

    /// Domains outside [0, 1] are sampled inside themselves (fixed 0.25..0.75 samples would evaluate to nothing on
    /// both surfaces and look alike).
    #[test]
    fn far_apart_surfaces_on_a_shifted_domain_do_not_coincide() {
        let mut a = wall_surface();
        let mut c = wall_surface();
        for s in [&mut a, &mut c] {
            s.u_knots = vec![10.0, 10.5, 11.0];
            s.v_knots = vec![10.0, 11.0];
        }
        assert!(surfaces_coincide(&a, &c));
        for pole in &mut c.poles {
            pole.y += 5.0;
        }
        assert!(!surfaces_coincide(&a, &c));
    }
}
