import makeRegl from "regl";


function simple_tri(regl) {
	return regl({
		frag: `
		uniform int w,h;
		void main() {
			gl_FragColor = vec4(gl_FragCoord.xy/vec2(w,h), 1., 1.);
		}`,
		vert: `
		attribute vec3 pos;
		void main() {
			gl_Position = vec4(pos, 1.);
		}`,
		uniforms: {
			w: (ctx,props) => ctx.viewportWidth,
			h: (ctx,props) => ctx.viewportHeight
		},
		attributes: {
			pos: [[-.85, -.85, 0], [.85, -.85, 0], [0, .85, 0]]
		},
		count: 3,
	});
}

export default class Scene {
	constructor(canvas) {
		console.log(canvas);
		this.regl = makeRegl(canvas);
		console.log('created regl');

		this.regl.clear({
			color: [0, 0, 0, 1],
			depth: 1,
			stencil: 0
		})

		let tri = simple_tri(this.regl);
		tri();
	}
}
