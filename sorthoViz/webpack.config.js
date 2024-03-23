// const path = require('path');
// const HtmlWebpackPlugin = require('html-webpack-plugin');
import path from 'path';
import HtmlWebpackPlugin from 'html-webpack-plugin';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
const __dirname = dirname(fileURLToPath(import.meta.url));

import { setupDummyMiddlewares } from './src/example_server/serve.js';

export default function(env, argv) {
	console.log('production build = ', env.production);

	return {
		name: 'index',
		entry: './src/index.js',
		mode: env.production ? 'production' : 'development',
		devtool: env.production ? false : 'inline-source-map',
		output: {
			filename: 'index.bundle.js',
			path: path.resolve(__dirname, 'dist'),
		},

		plugins: [new HtmlWebpackPlugin()],

		// watch: true,
		devServer: {
			static: {
				directory: path.join(__dirname, 'public'),
			},
			compress: true,
			port: 9001,
			client: {
				overlay: true,
			},

			watchFiles: ['**/*.js', 'webpack.config.js'],

			// NOTE: install our example server endpoints into the dev server
			setupMiddlewares: (middlewares, devServer) => {
				return setupDummyMiddlewares(middlewares, devServer);
			},
		},

		module: {
			rules: [

				// https://webpack.js.org/loaders/postcss-loader/
				{
					test: /\.css$/i,
					use: [
						"style-loader",
						"css-loader",
						{
							loader: "postcss-loader",
							options: {
								postcssOptions: {
									plugins: [
										[
											"postcss-preset-env",
											{
												// Options
											},
										],
									],
								},
							},
						},
					],
				},

			],
		},
	};
}
