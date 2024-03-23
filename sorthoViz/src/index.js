
import m from "mithril";
import "./main.css";

import Scene from "./threed/scene.js";

var root = document.body

let sceneComponent = {
	oncreate: function(vnode) {
		console.log('oncreate', vnode.dom);
		let canvas = vnode.dom;

		// WARNING: Should I really construct this here?
		vnode.state.scene = new Scene(canvas);

		/*
		m.request({method: 'GET', url: '/image/0'})
		 .then(function(result) {
			 sceneComponent.
		 });
		*/
	},
	view: function(vnode) {
		console.log('view');
		return m("canvas", {width: 1536, height: 1024})
	}
};

m.mount(root, {
	view: function() {
		return m('div', [
			m("h2", {class: 'mb-4 text-3xl font-extrabold text-slate-50 p-4'}, "Hello World"),
			m(sceneComponent),
		]);
	}
});
