import makeRegl from "regl";
import {FreeCamera} from "./camera/cam_free.js";
import {TileRenderer} from "./tileRenderer/renderer.js";


function make_simple_tri(regl) {
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

function make_simple_tri2(regl) {
	return regl({
		frag: `
		uniform int w,h;
		void main() {
			gl_FragColor = vec4(gl_FragCoord.xy/vec2(w,h), 1., 1.);
		}`,
		vert: `
		attribute vec3 pos;
		uniform mat4 view;
		uniform mat4 proj;
		void main() {
			gl_Position = proj * view * vec4(pos, 1.);
		}`,
		uniforms: {
			view: (ctx,props) => {
				// console.log('ctx', ctx);
				return ctx.camView;
			},
			proj: (ctx,props) => ctx.camProjection,
			w: (ctx,props) => ctx.viewportWidth,
			h: (ctx,props) => ctx.viewportHeight
		},
		attributes: {
			pos: [[-.85, -.85, 0], [.85, -.85, 0], [0, .85, 0]]
		},
		count: 3,
	});
}


// Convert props to context variables. Makes things easier later on.
function make_setup_cam(regl) {
	return regl({
		context: {
			camProjection: function(ctx,props) {
				return props.cam.proj;
			},
			camView: function(ctx,props) {
				return props.cam.view;
			},
		}
	});
}

export default class Scene {
	constructor(canvas) {
		canvas.tabIndex=-1;
		console.log(canvas);
		this.regl = makeRegl(canvas);
		console.log('created regl');

		this.regl.clear({
			color: [0, 0, 0, 1],
			depth: 1,
			stencil: 0
		})

		let tri = make_simple_tri(this.regl);
		let tri2 = make_simple_tri2(this.regl);

		this.cam = new FreeCamera(canvas);
		// console.log(this.cam);
		let setupCamera = make_setup_cam(this.regl);
		let self = this;

		let tileRenderer = new TileRenderer(this.regl);

		this.regl.frame(function(ctx) {

			self.cam.step(ctx);

			self.regl.clear({
				color: [0, 0, 0, 1],
				depth: 1,
				stencil: 0
			})

			setupCamera({
				cam: self.cam
			}, function(ctx) {
				tri2();
				tileRenderer.render();
			});
		});
	}
}
