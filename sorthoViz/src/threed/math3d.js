//
// NOTE: Mostly copied & edited from my row-major version.
// There may be bugs.
//

// ---------------------------------------------------------------
// Matrices
// ---------------------------------------------------------------

export function eye() {
	return new Float32Array([
		1,0,0,0,
		0,1,0,0,
		0,0,1,0,
		0,0,0,1]);
}

export function mtv(A, x) {
	return new Float32Array([
		A[0*4+0]*x[0] + A[0*4+1]*x[1] + A[0*4+2]*x[2] + A[0*4+3]*x[3],
		A[1*4+0]*x[0] + A[1*4+1]*x[1] + A[1*4+2]*x[2] + A[1*4+3]*x[3],
		A[2*4+0]*x[0] + A[2*4+1]*x[1] + A[2*4+2]*x[2] + A[2*4+3]*x[3],
		A[3*4+0]*x[0] + A[3*4+1]*x[1] + A[3*4+2]*x[2] + A[3*4+3]*x[3]]);
}

// Transpose the top-left 3x3 corner of A and multiply the 3-vector x
export function topLeft_mtv_3(A, x) {
	return new Float32Array([
		A[0*4+0]*x[0] + A[0*4+1]*x[1] + A[0*4+2]*x[2],
		A[1*4+0]*x[0] + A[1*4+1]*x[1] + A[1*4+2]*x[2],
		A[2*4+0]*x[0] + A[2*4+1]*x[1] + A[2*4+2]*x[2]]);
}

export function mv(A, x) {
	return new Float32Array([
		A[0*4+0]*x[0] + A[1*4+0]*x[1] + A[2*4+0]*x[2] + A[3*4+0]*x[3],
		A[0*4+1]*x[0] + A[1*4+1]*x[1] + A[2*4+1]*x[2] + A[3*4+1]*x[3],
		A[0*4+2]*x[0] + A[1*4+2]*x[1] + A[2*4+2]*x[2] + A[3*4+2]*x[3],
		A[0*4+3]*x[0] + A[1*4+3]*x[1] + A[2*4+3]*x[2] + A[3*4+3]*x[3]]);
}

export function mm(A, B) {
	let C = new Float32Array(16);
	for (let i=0; i<4; i++)
		for (let j=0; j<4; j++)
			for (let k=0; k<4; k++)
				C[j*4+i] += A[k*4+i]*B[j*4+k];
	return C;
}

export function gemm(A, B, L, I, R) {
	let C = new Float32Array(L*R);
	for (let i=0; i<L; i++)
		for (let j=0; j<R; j++)
			for (let k=0; k<I; k++)
				C[j*L+i] += A[k*I+i]*B[j*I+k];
	return C;
}
export function generalTranspose(A, L, R) {
	let B = new Float32Array(L*R);
	for (let i=0; i<L; i++) {
		for (let j=0; j<R; j++) {
			B[i*R+j] = A[i+j*L];
		}
	}
	return B;
}

export function inverseSE3(A) {
	const x = -(A[0*4+0]*A[3*4+0] + A[0*4+1]*A[3*4+1] + A[0*4+2]*A[3*4+2]);
	const y = -(A[1*4+0]*A[3*4+0] + A[1*4+1]*A[3*4+1] + A[1*4+2]*A[3*4+2]);
	const z = -(A[2*4+0]*A[3*4+0] + A[2*4+1]*A[3*4+1] + A[2*4+2]*A[3*4+2]);
	return new Float32Array([
		// A[0+4*0], A[1+4*0], A[2+4*0], 0,
		// A[0+4*1], A[1+4*1], A[2+4*1], 0,
		// A[0+4*2], A[1+4*2], A[2+4*2], 0,
		A[0*4+0], A[1*4+0], A[2*4+0], 0,
		A[0*4+1], A[1*4+1], A[2*4+1], 0,
		A[0*4+2], A[1*4+2], A[2*4+2], 0,
		x,y,z,1]);
}

export function inverseGeneral(A) {
	throw 'not implement'
}
export function inverseProj(A) {
	// Use sympy: https://live.sympy.org/
	//
	// a,b,c,d,e,f = symbols('a b c d e f')
	// P = Matrix([ [a,0,b,0], [0,c,d,0], [0,0,e,f], [0,0,1,0]])
	// # OR: Matrix([ [a,0,b,0], [0,c,d,0], [0,0,e,f], [0,0,-1,0]])
	// P.inv()
	//
	const a = A[0*4+0];
	const b = A[2*4+0];
	const c = A[1*4+1];
	const d = A[2*4+1];
	const e = A[2*4+2];
	const f = A[3*4+2];
	return new Float32Array([
		// 1/a, 0, 0, b/a,
		// 0, 1/c, 0, d/c,
		// 0, 0, 0, -1,
		// 0, 0, 1/f, e/f
		1/a, 0, 0, 0,
		0, 1/c, 0, 0,
		0, 0, 0, 1/f,
		-b/a, -d/c, 1, -e/f // for cam along +Z
		// b/a, d/c, -1, e/f // For cam along -Z
	]);
}

export function normalized(x) {
	const n = Math.sqrt(x.map(e=>e*e).reduce((a,b)=>a+b, 0));
	return x.slice().map(a=>a/n);
}
export function norm(x) {
	return Math.sqrt(x.map(e=>e*e).reduce((a,b)=>a+b, 0));
}

export function cross(x,y) {
	return new Float32Array([
		-x[2]*y[1] + x[1]*y[2],
		 x[2]*y[0] - x[0]*y[2],
		-x[1]*y[0] + x[0]*y[1]])
}

export function crossMatrix(x) {
	return new Float32Array([
		0,x[2],-x[1],
		-x[2],0, x[0],
		 x[1],-x[0],0]);
}

export function transpose(A) {
	// return A;
	return new Float32Array([
		A[0*4+0], A[1*4+0], A[2*4+0], A[3*4+0],
		A[0*4+1], A[1*4+1], A[2*4+1], A[3*4+1],
		A[0*4+2], A[1*4+2], A[2*4+2], A[3*4+2],
		A[0*4+3], A[1*4+3], A[2*4+3], A[3*4+3]]);
}

// ---------------------------------------------------------------
// Graphics
// ---------------------------------------------------------------

export function q_enu_from_rwf() {
	const a = 0.70710678118;
	return [a, -a, 0, 0];
}
export function R_enu_from_rwf() {
	return [
		1,0,0, 0,
		0,0,-1, 0, // column major!
		0,1,0, 0,
		0,0,0,1
	];
}
export function make_viewInv_enu(pq, sq, t) {
	let A = mm(R_enu_from_rwf() , qmatrix(qmult(pq,sq))); // world_from_body.
	A[12] = t[0];
	A[13] = t[1];
	A[14] = t[2];
	A[15] = 1;
	return A;
}
export function make_viewInv_ltp(pq, sq, t) {
	const diff = [t[0], t[1], t[2]];
	let f = normalized(diff);
	let r = normalized(cross([0,0,-1], f));
	// let r = normalized(cross(f,[0,0,1]));
	let u = normalized(cross(f,r));
	// let u = normalized(cross(r,f));
	const LTP = [
		r[0], r[1], r[2], 0,
		u[0], u[1], u[2], 0,
		f[0], f[1], f[2], 0,
		0,0,0,1
	];
	console.log(mm(LTP,transpose(LTP)))

	const A = mm(R_enu_from_rwf() , qmatrix(qmult(pq,sq)));
	let B = mm(LTP , A);

	B[12] = t[0];
	B[13] = t[1];
	B[14] = t[2];
	B[15] = 1;
	return B;
}

export function lookAt(eye, target, up) {
	const diff = [target[0]-eye[0], target[1]-eye[1], target[2]-eye[2]];
	let f = normalized(diff);
	// let r = normalized(cross(up, f));
	let r = normalized(cross(f,up));
	let u = normalized(cross(f,r));
	r = cross(u,f);
	return inverseSE3(new Float32Array([
		// r[0], u[0], f[0], 0,
		// r[1], u[1], f[1], 0,
		// r[2], u[2], f[2], 0,
		r[0], r[1], r[2], 0,
		u[0], u[1], u[2], 0,
		f[0], f[1], f[2], 0,
		eye[0], eye[1], eye[2], 1]));
}

export function frustum(n,f, l,r, t,b) {
	return transpose(
		new Float32Array([
		// 2*n/(r-l), 0, (r+l)/(r-l), 0,
		// 0, 2*n/(t-b), (t+b)/(t-b), 0,
		-2*n/(r-l), 0, (r+l)/(r-l), 0,
		0, -2*n/(t-b), (t+b)/(t-b), 0,
		0, 0,  (f+n)/(f-n),  -2*f*n/(f-n), // For cam +Z (not opengl default)
		0,0,1,0]));
		// 2*n/(r-l), 0, (r+l)/(r-l), 0,
		// 0, 2*n/(t-b), (t+b)/(t-b), 0,
		// 0, 0,  -(f+n)/(f-n),  -2*f*n/(f-n), // For cam -Z (opengl default, but unintuitive)
		// 0,0,-1,0]));
}
export function frustumFromIntrin(n,f, intrin) {
	throw 'not implement'
}

// ---------------------------------------------------------------
// Quaternions
// ---------------------------------------------------------------

export function qmult(p,q) {
	const a1=p[0], b1=p[1], c1=p[2], d1=p[3];
	const a2=q[0], b2=q[1], c2=q[2], d2=q[3];
	return new Float32Array([
            a1*a2 - b1*b2 - c1*c2 - d1*d2,
            a1*b2 + b1*a2 + c1*d2 - d1*c2,
            a1*c2 - b1*d2 + c1*a2 + d1*b2,
            a1*d2 + b1*c2 - c1*b2 + d1*a2]);
}
export function qexp(u) {
	let n = Math.sqrt(u.map(e=>e*e).reduce((a,b)=>a+b, 0));
	if (n < 1e-16) {
		return new Float32Array([1,0,0,0]);
	}
	let k = u.slice().map(a=>a/n);
	return new Float32Array([
		Math.cos(n*.5),
		Math.sin(n*.5) * k[0],
		Math.sin(n*.5) * k[1],
		Math.sin(n*.5) * k[2]]);
}
export function qlog(u) {
	throw 'not implement'
}
export function qmatrix(u) {
	const q0=u[0], q1=u[1], q2=u[2], q3=u[3];
	return transpose(new Float32Array([
        q0*q0+q1*q1-q2*q2-q3*q3, 2*(q1*q2-q0*q3), 2*(q0*q2+q1*q3), 0,
        2*(q1*q2+q0*q3), (q0*q0-q1*q1+q2*q2-q3*q3), 2*(q2*q3-q0*q1), 0,
        2*(q1*q3-q0*q2), 2*(q0*q1+q2*q3), q0*q0-q1*q1-q2*q2+q3*q3, 0,
		0,0,0,1
	]));
}

export function qFromMatrix4(A) {
	if (A.length != 16) throw Error('bad input size.');

	/*
	// Based on this Eigen code:
    Scalar t = mat.trace();
    if (t > Scalar(0))
    {
      t = sqrt(t + Scalar(1.0));
      q.w() = Scalar(0.5)*t;
      t = Scalar(0.5)/t;
      q.x() = (mat.coeff(2,1) - mat.coeff(1,2)) * t;
      q.y() = (mat.coeff(0,2) - mat.coeff(2,0)) * t;
      q.z() = (mat.coeff(1,0) - mat.coeff(0,1)) * t;
    }
    else
    {
      Index i = 0;
      if (mat.coeff(1,1) > mat.coeff(0,0))
        i = 1;
      if (mat.coeff(2,2) > mat.coeff(i,i))
        i = 2;
      Index j = (i+1)%3;
      Index k = (j+1)%3;

      t = sqrt(mat.coeff(i,i)-mat.coeff(j,j)-mat.coeff(k,k) + Scalar(1.0));
      q.coeffs().coeffRef(i) = Scalar(0.5) * t;
      t = Scalar(0.5)/t;
      q.w() = (mat.coeff(k,j)-mat.coeff(j,k))*t;
      q.coeffs().coeffRef(j) = (mat.coeff(j,i)+mat.coeff(i,j))*t;
      q.coeffs().coeffRef(k) = (mat.coeff(k,i)+mat.coeff(i,k))*t;
    }
	*/

	let q = new Float32Array(4);
	let t = A[0] + A[1*4+1] + A[2*4+2];
	if (t > 0) {
		t = Math.sqrt(t+1);
		q[0] = .5*t;
		t = .5/t;
		q[1] = (A[1*4+2] - A[2*4+1]) * t;
		q[2] = (A[2*4+0] - A[0*4+2]) * t;
		q[3] = (A[0*4+1] - A[1*4+0]) * t;
	} else {
		let i = 0;
		if (A[1*4+1] > A[0]) i = 1;
		if (A[2*4+2] > A[i*4+i]) i = 2;
		let j = (i+1)%3;
		let k = (j+1)%3;
		t = Math.sqrt(A[i*4+i]-A[j*4+j]-A[k*4+k] + 1);
		q[1+i] = .5 * t;
		t = .5/t;
		q[0] = (A[j*4+k]-A[k*4+j])*t;
		q[1+j] = (A[i*4+j]+A[j*4+i])*t;
		q[1+k] = (A[i*4+k]+A[k*4+i])*t;
	}
	return q;
}

// ---------------------------------------------------------------
// Geographic conversions
// ---------------------------------------------------------------

const Earth = (() => {
	const R1         = (6378137.0);
	const R2         = (6356752.314245179);
	const a          = 1;
	const b          = R2 / R1;
	return {
		R1         : (6378137.0),
		R2         : (6356752.314245179),
		a          : 1,
		b          : R2 / R1,
		b2_over_a2 : (b*b) / (a*a),
		e2         : 1 - (b * b / a * a),
	};
})();

export function uwm_to_geodetic(x) {
	return new Float32Array([
		x[0] * Math.PI,
		Math.atan(Math.exp(x[1] * Math.PI)) * 2 - Math.PI / 2,
		x[2] * Math.PI]);
}
export function uwm_to_ecef(x) {
	return geodetic_to_ecef(uwm_to_geodetic(x));
}
export function geodetic_to_unit_wm(x) {
	return new Float32Array([
		x[0] / Math.PI,
		Math.log(Math.tan(Math.PI/4 + x[1]*.5)) / Math.PI,
		x[2] / Math.PI
	]);
}
export function ecef_to_geodetic(ecef) {
	const x = ecef[0], y = ecef[1], z = ecef[2];
	const ox = Math.atan2(y,x);
	let   k = 1. / (1. - Earth.e2);
	const p2 = x*x + y*y;
	const p = Math.sqrt(p2);
	for (let i=0; i<2; i++) {
		const c = Math.pow(((1-Earth.e2) * z*z) * (k*k) + p2, 1.5) / Earth.e2;
		k = (c + (1-Earth.e2) * z * z * Math.pow(k, 3)) / (c - p2);
	}
	const oy = Math.atan2(k*z, p);

	const rn = Earth.a / Math.sqrt(1-Earth.e2 * Math.pow(Math.sin(oy), 2));
	const sinabslat = Math.sin(Math.abs(oy));
	const coslat = Math.cos(oy);
	const oz = (Math.abs(z) + p - rn * (coslat + (1-Earth.e2) * sinabslat)) / (coslat + sinabslat);

	return new Float32Array([ ox, oy, oz ]);
}
export function ecef_to_uwm(x) {
	return geodetic_to_unit_wm(ecef_to_geodetic(x));
}
export function geodetic_to_ecef(g) {
	const cp = Math.cos(g[1]);
	const sp = Math.sin(g[1]);
	const cl = Math.cos(g[0]);
	const sl = Math.sin(g[0]);
	const n_phi = Earth.a / Math.sqrt(1 - Earth.e2 * sp * sp);
	return new Float32Array([
		(n_phi + g[2]) * cp * cl,
		(n_phi + g[2]) * cp * sl,
		(Earth.b2_over_a2 * n_phi + g[2]) * sp
	]);
}


// ---------------------------------------------------------------
// Higher level export functions
// ---------------------------------------------------------------


export function make_proj_ivew(state, cam, n, f) {
	if (n === undefined) n = 10/6e6;
	if (f === undefined) f = 1220/6e6;

	const t  = state.pos;
	const pq = state.pq;
	const sq = state.sq;
	// Buggy transpiler fails with this.
	// const [fx,fy] = cam.f;
	// const [cx,cy] = cam.c;
	// const [w,h] = cam.size;
	const fx = cam.f[0];
	const fy = cam.f[1];
	const cx = cam.c[0];
	const cy = cam.c[1];
	const w = cam.size[0];
	const h = cam.size[1];
	let u = .5*n*w/fx;
	let v = .5*n*h/fy;
	// let proj = frustum(n,f, -u,u, v,-v);
	// let proj = frustum(n,f, -u,u, -v,v);
	let proj = frustum(n,f, u,-u, v,-v);
	let ivew = make_viewInv_ltp(pq, sq, t);

	return [proj, ivew];
}

