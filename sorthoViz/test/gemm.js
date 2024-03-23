// NOTE: using import requires (no pun intended) putting 'type': 'module' in package.json
import 'mocha';
import assert from "assert";

// import math from "../src/threed/math3d.js";
import {gemm} from '../src/threed/math3d.js'

describe('gemm', function() {
	it('identity should be identity', function() {

		const A = [1,0,0, 0, 1, 0, 0, 0, 1];
		const B = [1,2,3,4,5,6];
		const C = gemm(A,B, 3, 3, 2);


		assert(C[0] == 1);
		assert(C[1] == 2);
		assert(C[2] == 3);
		assert(C[3] == 4);
		assert(C[4] == 5);
		assert(C[5] == 6);
	});

	it('should be correct', function() {
		//
		// A = 1 0 0
		//     0 0 0
		//     0 1 1
		//
		// B = 1 4
		//     2 5
		//     3 6
		//
		// C = 1 4
		//     0 0
		//     5 11
		//
		const A = [1,0,0, 0, 0, 1, 0, 0, 1];
		const B = [1,2,3,4,5,6];
		const C = gemm(A,B, 3, 3, 2);

		// console.log(A);
		// console.log(B);
		// console.log(C);


		assert(C[0] == 1);
		assert(C[1] == 0);
		assert(C[2] == 5);
		assert(C[3] == 4);
		assert(C[4] == 0);
		assert(C[5] == 11);
	});
});
