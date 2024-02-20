
def format_size(n):
    if n > (1<<30): return f'{n/(1<<30):2.1f}GB'
    if n > (1<<20): return f'{n/(1<<20):2.1f}MB'
    if n > (1<<10): return f'{n/(1<<10):2.1f}KB'
    return f'{n}B'
