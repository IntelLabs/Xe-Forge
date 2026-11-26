"""x-forge-skill analyze: AST-based PyTorch kernel analysis."""


def run(args):
    from x_forge.core.kernel_analyzer import KernelAnalyzer, format_analysis

    analyzer = KernelAnalyzer()
    result = analyzer.analyze(args.pytorch_file)
    print(format_analysis(result, name=args.pytorch_file))
