using DeepCompare.Engine;

namespace DeepCompare.App;

/// <summary>選べる符号化 1 つ。画面には <see cref="Label"/> を出す。</summary>
public sealed record EncodingChoice(TextEncoding Value, string Label);
