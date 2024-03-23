import {gemm, norm} from '../math3d.js'


function make_render_tile_cmd(regl) {
}


class Bbox {
	constructor(loHi) {
		if (loHi.length != 6) throw new Error("invalid loHi passed to Bbox");
		this.loHi = loHi;

		this.corners_4x8 = [
			loHi[0*3+0], loHi[0*3+1], loHi[0*3+2], 1,
			loHi[0*3+0], loHi[0*3+1], loHi[1*3+2], 1,
			loHi[0*3+0], loHi[1*3+1], loHi[1*3+2], 1,
			loHi[0*3+0], loHi[1*3+1], loHi[0*3+2], 1,
			loHi[1*3+0], loHi[0*3+1], loHi[0*3+2], 1,
			loHi[1*3+0], loHi[0*3+1], loHi[1*3+2], 1,
			loHi[1*3+0], loHi[1*3+1], loHi[1*3+2], 1,
			loHi[1*3+0], loHi[1*3+1], loHi[0*3+2], 1 ];
	}

	corners() {
		return this.corners_84_;
	}

	projectCorners(mvp) {
		let projCorners = gemm(mvp, this.corners_4x8, 4, 4, 8);
		for (let j=0; j<3; j++)
			for (let i=0; i<8; i++)
				projCorners[j*8+i] = projCorners[j*8+i] / projCorners[3*8+i];
		console.log('projCorners',projCorners);
	}

}

// NOTE: My first impl will be synchronous

class Tile {
	constructor(regl, verts, inds, baseTex, modelMatrix) {
		if (Tile.prototype.draw === undefined) Tile.prototype.cmd = make_Tile_command(regl);
		this.verts = regl.buffer(verts);
		this.elements = regl.elements({primitive: 'triangles', usage: 'static', data: inds})
		this.baseTex = baseTex;
		this.modelMatrix = modelMatrix
	}
}

function make_Tile_command(regl) {
	return regl({
		frag: `precision mediump float;
		uniform int w,h;
		varying vec2 v_uv;
		uniform sampler2D tex;
		void main() {
			gl_FragColor = texture2D(tex, v_uv);
		}`,
		vert: `precision mediump float;
		attribute vec4 a_pos;
		attribute vec2 a_uv;
		varying vec2 v_uv;
		uniform mat4 view;
		uniform mat4 proj;
		uniform mat4 model;
		void main() {
			gl_Position = model * proj * view * vec4(a_pos.xyz, 1.);
			v_uv = a_uv;
		}`,
		uniforms: {
			model: regl.this('modelMatrix'),
			view: (ctx,props) => {
				// console.log('ctx', ctx);
				return ctx.camView;
			},
			proj: (ctx,props) => ctx.camProjection,
			w: (ctx,props) => ctx.viewportWidth,
			h: (ctx,props) => ctx.viewportHeight,
			tex: regl.this('baseTex'),
		},
		attributes: {
			a_pos: {
				buffer: regl.this('verts'),
				offset: 0,
				stride: 24,
				size: 4,
			},
			a_uv: {
				buffer: regl.this('verts'),
				offset: 16,
				stride: 24,
				size: 2,
			},
		},
		elements: regl.this('elements'),
	});
}


const CLOSED = 0;
const OPENING = 1;
const OPEN = 2;
const CLOSING = 3;

class TileNode {
	constructor(id, bbox, geoError) {
		this.id = id;
		this.bbox = bbox;
		this.children = [];
		this.geoError = geoError;
		this.state = CLOSED;
	}

	// The error in pixels incurred for _not_ opening this node
	computeSse(mvp) {
	}

	open() {
	}

	close() {
	}

}

export class TileRenderer {
	constructor(regl) {
		this.regl = regl;

		let testVerts = [
			0,0,0,1, 0,0,
			1,0,0,1, 1,0,
			1,1,0,1, 1,1,
			0,1,0,1, 0,1];
		let testInds = [0,1,2, 2,3,0];
		let testTexData = [];
		for (let i=0; i<256; i++) {
			for (let j=0; j<256; j++) {
				testTexData.push(0);
				testTexData.push(i);
				testTexData.push(j);
				testTexData.push(200);
			}
		}
		let testTex = regl.texture({
			shape: [256,256],
			type: 'uint8',
			format: 'rgba',
			data: testTexData});
		let testModelMatrix = [
			1,0,0,0,
			0,1,0,0,
			0,0,1,0,
			0,0,0,1];
		this.testTile = new Tile(regl, testVerts, testInds, testTex, testModelMatrix);
	}

	render(props) {
		this.testTile.cmd(props)
	}
}
