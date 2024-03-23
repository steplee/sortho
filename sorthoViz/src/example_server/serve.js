export function setupDummyMiddlewares(middlewares, devServer) {
	devServer.app.get('/hello',	(req,res) => {
		res.send('hello world 1');
	});
	devServer.app.get('/image/:id',	(req,res) => {
		res.send(`got image id=${req.params.id}, all params = ${JSON.stringify(req.params)}, all queries = ${JSON.stringify(req.query)}`);
	});
	middlewares.push((req, res) => {
		res.send('Not a valid path');
	});
	return middlewares;
}
