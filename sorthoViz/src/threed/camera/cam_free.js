import {frustum, qmatrix, inverseSE3, qexp, qmult, topLeft_mtv_3, norm} from '../math3d.js'

// To be used with https://github.com/mikolalysenko/mouse-change/blob/master/mouse-listen.js
export function FreeCamera(element) {
	this.fov = 45;
	this.q = new Float32Array([1,0,0,0]);
	this.t = new Float32Array([0,0,-1.1]);
	this.v = new Float32Array([0,0,0]);
	this.computeMatrices({viewportWidth:1, viewportHeight:1});

	this.lastX = undefined;
	this.lastY = undefined;
	this.lastTime = undefined;
	this.leftDown = false;
	this.rghtDown = false;
	this.dx = 0;
	this.dy = 0;
	this.shift = false;
	this.alt = false;
	this.keys = [];

	// element.addEventListener('mousemove', this.handleMouseMove)
	// element.addEventListener('mousedown', this.handleMouseDown)
	// element.addEventListener('mouseup', this.handleMouseUp)
	element.addEventListener('mouseup', this.handleMouse.bind(this))
	element.addEventListener('mousedown', this.handleMouse.bind(this))
	element.addEventListener('mousemove', this.handleMouse.bind(this))
	element.addEventListener('mouseleave', this.handleClear.bind(this))
	element.addEventListener('mouseenter', this.handleClear.bind(this))
	element.addEventListener('mouseover', this.handleClear.bind(this))
	element.addEventListener('mouseout', this.handleClear.bind(this))
	element.addEventListener('keyup', this.handleKeyUp.bind(this))
	element.addEventListener('keydown', this.handleKeyDown.bind(this))
	// element.addEventListener('keypress', this.handleKey)
	this.element = element;
	// this.element.camera = this;
	// window.camera_ = this;

}
FreeCamera.prototype.destroy = function() {
	// element.removeEventListener('mousemove', this.handleMouseMove)
	// element.removeEventListener('mousedown', this.handleMouseDown)
	// element.removeEventListener('mouseup', this.handleMouseUp)
	let element = this.element;
	element.removeEventListener('mouseleave', this.handleClear)
	element.removeEventListener('mouseenter', this.handleClear)
	element.removeEventListener('mouseover', this.handleClear)
	element.removeEventListener('mouseout', this.handleClear)
	element.removeEventListener('keyup', this.handleKeyUp)
	element.removeEventListener('keydown', this.handleKeyDown)
	// element.removeEventListener('keypress', this.handleKey)
}
FreeCamera.prototype.handleMouse = function(e) {
	// console.log('mouse', e, 'this', this);
	this.leftWasDown = this.leftDown;
	this.rghtWasDown = this.rghtDown;
	this.leftDown = e.buttons & 1;
	this.rghtDown = e.buttons & 2;
	if (this.lastX == undefined) {
		this.dx = 0;
		this.dy = 0;
	} else {
		this.dx = this.lastX - e.offsetX;
		this.dy = this.lastY - e.offsetY;
	}
	// console.log(this.lastX, "=>",this.dx, this.dy);
	// this.dx = e.movementX;
	// this.dy = e.movementY;
	this.lastX = e.offsetX;
	this.lastY = e.offsetY;
	// console.log(this, this.dx, this.lastX,this.keys);
}
FreeCamera.prototype.handleClear = function(e) {
	this.lastX = undefined;
	this.lastY = undefined;
	this.lastTime = undefined;
	this.leftDown = false;
	this.rghtDown = false;
	this.dx = 0;
	this.dy = 0;
	this.shift = this.alt = false;
	this.keys = [];
}
FreeCamera.prototype.handleKeyUp = function(e) {
	if (e.key == "Shift") {
		this.shift = false;
		return;
	}
	if (e.key == "Alt") {
		this.alt = false;
		return;
	}

	let idx = this.keys.indexOf(e.key);
	if (idx >= 0) {
		this.keys.splice(idx,1);
	}
}
FreeCamera.prototype.handleKeyDown = function(e) {
	console.log(e);
	if (e.key == "Shift") {
		this.shift = true;
		return;
	}
	if (e.key == "Alt") {
		this.alt = true;
		return;
	}

	if (!e.repeat) {
		let idx = this.keys.indexOf(e.key);
		if (idx == -1)
			this.keys.push(e.key);
	}
}

FreeCamera.prototype.computeMatrices = function(ctx) {
	if (this.lastViewport != [ctx.viewportWidth, ctx.viewportHeight]) {
		this.lastViewport = [ctx.viewportWidth, ctx.viewportHeight];
		const n = .01;
		// const s = Math.tan(this.fov*.5*3.141/180);
		const s = .7;
		let u = n * s * (ctx.viewportWidth / ctx.viewportHeight), v = n * s;
		this.proj = frustum(n, 4.5, -u,u, -v,v);
	}

	let VI = qmatrix(this.q);
	VI[3*4+0] = this.t[0];
	VI[3*4+1] = this.t[1];
	VI[3*4+2] = this.t[2];
	VI[3*4+3] = 1;
	// console.log(VI);
	// this.VI = VI;
	this.view = inverseSE3(VI);
}
FreeCamera.prototype.step = function(ctx) {
	// let {dx, dy, down} = this.updateIo();
	// this.updateIo();
	const dx = this.dx, dy = this.dy;
	const dt = 1/60.;

	if (this.leftDown) {
		const qspeed = .3 * dt;
		let dq1 = qexp([-this.dy*qspeed,0,0]);
		let dq2 = qexp([0,0,this.dx*qspeed]);
		this.q = qmult(dq2, qmult(this.q, dq1));
	}

	let t = this.t;
	let v = this.v;

	let a = [
		(this.keys.indexOf("d") != -1) ?  1 : (this.keys.indexOf("a") != -1) ? -1 : 0,
		(this.keys.indexOf("q") != -1) ? -1 : (this.keys.indexOf("e") != -1) ?  1 : 0,
		(this.keys.indexOf("w") != -1) ?  1 : (this.keys.indexOf("s") != -1) ? -1 : 0,
	];
	a = topLeft_mtv_3(this.view, a);

	if (1) {
		let drag_= 289.5 * norm(v);
		let drag = Math.min(.99, drag_);
		a[0] -= drag * v[0];
		a[1] -= drag * v[1];
		a[2] -= drag * v[2];
	} else {
		let drag_ = 20.1 * dt;
		a[0] -= drag_ * v[0] * Math.abs(v[0]);
		a[1] -= drag_ * v[1] * Math.abs(v[1]);
		a[2] -= drag_ * v[2] * Math.abs(v[2]);
	}

	v[0] = v[0] + a[0]*dt;
	v[1] = v[1] + a[1]*dt;
	v[2] = v[2] + a[2]*dt;
	// console.log(v, this.keys)

	t[0] = t[0] + v[0]*dt;
	t[1] = t[1] + v[1]*dt;
	t[2] = t[2] + v[2]*dt;

	this.computeMatrices(ctx);
}

