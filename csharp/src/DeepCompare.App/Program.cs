using Avalonia;

namespace DeepCompare.App;

internal static class Program
{
    private const string Usage = """
        DeepCompare - 意味的な類似度で行を対応付けるコード比較ツール

          deepcompare [左 右]        GUI を開く（引数 2 つならそのまま比較する）
          deepcompare --structured 左 右
                                     GUI を開き、構造として比較する画面を出す
          deepcompare --git [場所]   GUI を開き、Git の画面を出す
          deepcompare --version-view 左 右
                                     GUI を開き、版の比較の画面を出す
          deepcompare --snapshot-view <フォルダー> [写し]
                                     GUI を開き、写しの画面を出す。写しも渡すと
                                     そのまま比べる
          deepcompare --merge-view 祖先 左 右
                                     GUI を開き、3 方向マージの画面を出す
          deepcompare --print 左 右  画面を開かず、比較結果をテキストで出す
          deepcompare --print-folder 左 右
                                     フォルダーを比較して一覧を出す。終了コードは
                                     0 差異なし / 1 差異あり / 2 異常（CI で使える）。
                                     zip / tar / tar.gz も中身をフォルダーとして扱う
          deepcompare --merge <向き> 左 右
                                     差分を反映する。向きは to-right か to-left。
                                     -o か --in-place を付けない限り何も書かない
          deepcompare --print-table 左 右
                                     CSV/TSV を列単位で比較する
          deepcompare --print-json 左 右
                                     JSON / XML / TOML / YAML を**構造として**
                                     比較する。キーの順序や整形の違いは差分に
                                     しない。形式は拡張子から決める。終了コードは
                                     0 差異なし / 1 差異あり / 2 異常
          deepcompare --merge3 祖先 左 右
                                     3 方向マージ。競合は <<<<<<< で囲む。
                                     終了コードは 0 競合なし / 1 競合あり
          deepcompare --git-status [場所]
                                     作業ツリーの状態。終了コードは
                                     0 きれい / 1 変更あり / 2 異常
          deepcompare --git-log [場所]
                                     履歴を一覧にする
          deepcompare --git-diff <ファイル>
                                     ある時点と今の中身を比べる。**git 本体の diff
                                     ではなく、こちらの比較エンジンに掛ける**ので、
                                     意味的な行の対応付けが効く
          deepcompare --git-resolve <ファイル>
                                     競合を 3 方向マージで解く。索引に積まれた
                                     祖先・こちら・むこうを使う（**作業ツリーの
                                     ファイルは読まない**。git が書いた印が
                                     混ざっている）。解決できたら索引へ載せる。
                                     決められない箇所が残ったら中身を出して止まる。
                                     --take-ours / --take-theirs でその側に寄せる。
                                     0 解決 / 1 競合が残る / 2 異常
          deepcompare --print-office 文書 [文書]
                                     .docx / .xlsx / .pptx の**本文**を出す。
                                     2 つ渡せば比べる。実体は zip + XML なので、
                                     中身をそのまま比べると書式や ID の
                                     書き換えで無関係な差分が大量に出る
                                     （開いて保存し直すだけでも動く）。
                                     0 差異なし / 1 差異あり / 2 異常
          deepcompare --print-notebook 左.ipynb 右.ipynb
                                     ノートブックを**セル単位で**比べる。
                                     行で比べると、実行しただけで出力の
                                     base64 が数千行動き、直した 1 行が
                                     その中に埋もれる。既定では出力と実行回数を
                                     見ない。--with-outputs / --with-execution-count
                                     で見る。--all で変化のないセルも出す。
                                     0 本文に変化なし / 1 変わった / 2 異常
          deepcompare --strip-notebook <ファイル.ipynb>
                                     出力と実行回数を落とす（nbstripout 相当）。
                                     git に入れる前に通しておけば、実行しただけで
                                     差分が出ることがなくなる。--in-place で上書き
          deepcompare --print-version-info 実行ファイル [実行ファイル]
                                     Windows の実行ファイル（PE）の版・会社名・
                                     説明などを出す。2 つ渡せば比べる。
                                     **Windows の API に頼らず自前で読む**ので、
                                     Linux 上でも Windows 向けの成果物を確かめ
                                     られる。--all で同じ項目も出す。
                                     0 同じ / 1 違う / 2 異常
          deepcompare --snapshot <フォルダー>
                                     いまの状態を写し取る（BC の Snapshot）。
                                     標準出力へ出すので `-o` かリダイレクトで
                                     残す。中身は保存しない — 名前・大きさ・
                                     時刻だけ。--hash を付けると指紋も取る
                                     （**同じ秒の内の書き換えは指紋でしか
                                     見分けられない**）
          deepcompare --snapshot-diff 写し [写し]
                                     写しと今の姿を比べる。2 つ目を渡せば
                                     写し同士。--all で変化のないものも出す。
                                     0 変化なし / 1 変化あり / 2 異常
          deepcompare --print-image 左 右
                                     画像を画素で比べる。大きさが違っても
                                     重なる範囲を比べ、はみ出した分は
                                     「片方だけ」とする。終了コードは
                                     0 同じ / 1 違う / 2 異常。**しきい値の
                                     内側の差だけなら 0**（JPEG を保存し直した
                                     だけの違いで CI が赤くなるのを避ける）
                                     --tolerance <0-255> 既定 8
                                     --ignore-alpha    透明度は見ない
          deepcompare --print-binary 左 右
                                     バイト列として 16 進で比較する。テキストとして
                                     読めないファイル向け。終了コードは
                                     0 同じ / 1 違う / 2 異常
          deepcompare --sync 左 右   フォルダーを揃える。**既定では予定を出すだけ。**
                                     実行するには --apply を付ける
          deepcompare --multi A B C …
                                     3 つ以上を並べて比べる（dev / staging / prod の
                                     設定など）。**どれが仲間外れか**を出す。
                                     --all を付けると同じものも並べる
          deepcompare --deps 左 右   依存の一覧（ロックファイル）の変化だけを出す。
                                     数千行動いても、意味のある変化は数行
          deepcompare --secrets <ファイル> [変更後]
                                     秘密（API キー・トークン・秘密鍵）が混ざって
                                     いないか調べる。2 つ渡すと**増えた行だけ**を
                                     見る。終了コードは 0 無い / 1 見つかった / 2 異常
          deepcompare --invisible <ファイル>
                                     「同じに見えるのに一致しない」原因を調べる。
                                     ゼロ幅文字・全角空白・ノーブレークスペース・
                                     Unicode 正規化・行末空白・改行の混在・BOM。
                                     終了コードは 0 何も無い / 1 見つかった / 2 異常
          deepcompare --font-check   日本語を表示できる書体があるかを調べる

        オプション
          -o <パス>        結果をファイルへ書く。Windows の GUI 版 exe は標準出力が
                           繋がらないため、遠隔から結果を回収するときはこれを使う
          --threshold <値> 対応とみなす類似度の下限（既定 0.50）
          --structural     埋め込みを使わず文字列一致だけで対応付ける。GUI が最初に
                           描くのと同じ内容で、桁違いに速い
          --link 左行:右行 その 2 行を対応させる（**行番号は 1 始まり**）。
                           複数回指定できる。自動の対応付けが外したところを直す
          --unlink 左行:右行
                           その 2 行の対応を外す
          --model <パス>   使うモデル。名前だけなら実行ファイルの隣を探す。
                           環境変数 DEEPCOMPARE_MODEL でも指定できる
                           （**入れ替えて試すのに再配置が要らない**）
          -h, --help       この説明

        重要でない差分（指定した違いは一致として扱う）
          --ws <方式>      空白の扱い。respect（既定）/ trailing（行末）/
                           ends（行頭行末）/ collapse（連続空白を 1 つ）/ all（全部）
          -i, --ignore-case   大文字小文字を区別しない
          --normalize-unicode
                           Unicode 正規化を揃えてから比べる（NFC に寄せる）。
                           macOS が作ったファイルは「が」が 2 文字で書かれている
                           ことがあり、表示は同じなのに一致しない。
                           **既定は切ってある。** 揃えると、正規化の違いそのものを
                           直したい場面で差分が見えなくなるため。
                           見つけたいときは --invisible を使う
          --ignore-pattern <正規表現>
                           一致した部分を比較から除く。複数回指定できる。
                           例: --ignore-pattern '\d{4}-\d{2}-\d{2}'

        表形式の比較のオプション（--print-table と併用）
          --key <列>       行の対応付けに使う列。番号（1 始まり）か見出し名。
                           複数回指定可。**並び順が違っても照合できる**
          --ignore-column <列>
                           比較から外す列。複数回指定可
          --delimiter <字> 区切り文字。tab も指定できる（既定は拡張子から推定）
          --no-header      1 行目を見出しとして扱わない

        Git のオプション
          --limit <数>     --git-log で出す件数（既定 50）
          --rev <revision> 対象の時点。--git-log と --git-diff で使う
                           （--git-diff の既定は HEAD）
          --path <パス>    --git-log をそのファイルに絞る
          --changes-only   --git-status で変化のない項目を出さない

        構造化データの比較のオプション（--print-json と併用）
          --array-key <名前>
                           配列の要素を対応付ける名前（既定は id / name / key / path）。
                           複数回指定可。**並び順が違っても照合できる**
          --ignore-path <位置>
                           比較しない位置。$.metadata.generated_at のように書く。
                           * で 1 段ぶんの任意に当たる（$.cells[*].execution_count）。
                           複数回指定可
          --ignore-order   配列の並び順の違いを報告しない
          --strict-numbers 1.0 と 1 を別のものとして扱う

        同期のオプション（--sync と併用）
          --direction <向き>
                           to-right（既定）/ to-left / both（新しい方を採る）
          --delete-orphans 片側にしか無いものを消す。**既定は消さない。**
                           同期の事故はほとんどこれで起きる
          --apply          実際に実行する。付けなければ予定を出すだけ
          --tolerance <秒> both のときの時刻の許容誤差（既定 2 秒）

        秘密の検出のオプション（--secrets と併用）
          --secret-level <段階>
                           どこから報告するか。high（形が決まっているものだけ）/
                           medium（既定。名前で拾ったものまで）/
                           low（乱数のような文字列まで）

        書き出しのオプション
          --report <形式>  unified（patch で適用できる差分）か html（左右並記）。
                           --print と併用。-o でファイルへ
          --context <行数> unified で変更の前後に付ける文脈（既定 3）
          --csv            --print-folder の結果を CSV で出す

        差分の反映のオプション（--merge と併用）
          --block <番号>   反映する塊を番号で指定。複数回指定可。既定は全部
          --in-place       写される側のファイルを直接書き換える
          -o <パス>        結果を別のファイルへ書く（元は変えない）

        フォルダー比較のオプション（--print-folder と併用）
          --include <型>   対象にする名前。* と ? が使える。複数回指定可。
                           ファイルにだけ効く（ディレクトリの走査は止めない）
          --exclude <型>   除外する名前。複数回指定可
          --by-timestamp   中身を読まず、大きさと更新時刻で比べる。数が多いとき用
          --tolerance <秒> 更新時刻の差をどこまで同じとみなすか
          --ignore-dst     ちょうど 1 時間のずれを同じとみなす（夏時間）
          --min-size <数>  この大きさ未満のファイルを対象から外す
          --max-size <数>  この大きさを超えるファイルを対象から外す
          --no-recurse     直下だけを比べる
          --changes-only   一致した項目を一覧に出さない
          --case-sensitive-names
                           ファイル名の大小文字を区別する（既定は区別しない）
          --detect-renames 名前が変わっただけのファイルを見つけて併記する。
                           BC は名前一致しか見ないので、ここは差になる部分

        """;

    [STAThread]
    public static int Main(string[] args)
    {
        // 画面を開かない経路を先に捌く。GUI を初期化する前に分岐しないと、
        // 表示できない環境で使えなくなる。
        if (Cli.TryRun(args, Usage) is { } exitCode)
        {
            return exitCode;
        }

        App.StartupFiles = Cli.Positional(args);
        App.StartStructured = args.Contains("--structured");
        App.StartGit = args.Contains("--git");
        App.StartMerge = args.Contains("--merge-view");
        App.StartVersion = args.Contains("--version-view");
        App.StartSnapshot = args.Contains("--snapshot-view");
        return BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);
    }

    public static AppBuilder BuildAvaloniaApp()
        => AppBuilder.Configure<App>()
            .UsePlatformDetect()
            .WithInterFont()
            .LogToTrace();
}
