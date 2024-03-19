const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');

module.exports = function(env, argv) {
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
				return require('./src/example_server/serve.js').setupDummyMiddlewares(middlewares, devServer);
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
