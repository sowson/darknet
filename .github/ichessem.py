import sys

if len(sys.argv) != 2:
    print("Usage: python3 chessem.py <sha256>")
    sys.exit(1)

sha256 = sys.argv[1]

formula = f"""class Ichess < Formula
  desc "iChess.io UCI-compatible chess engine with Chess960 and GPU acceleration"
  homepage "https://ichess.io"
  url "https://github.com/sowson/darknet/releases/download/ichess-v{{{{version}}}}/iChess.io.en.mac.zip"
  sha256 "{sha256}"
  license "MIT"
  version ENV["ICHESS_VERSION"] || "1.0"

  depends_on "clblas"
  depends_on "opencv"

  def install
    bin.install "iChess.io.en.mac" => "ichess"
  end

  def caveats
    <<~EOS
      The iChess engine has been installed as `ichess`.

      You can use it with any UCI-compatible chess GUI:
        - CuteChess
        - Scid vs PC
        - Banksia GUI

      Supports:
        - Classic & Fischer 960
        - GPU acceleration (OpenCL/clBLAS)
        - Self-learning via UCI
    EOS
  end

  test do
    output = pipe_output("\#{bin}/ichess", "uci\\n", 2)
    assert_match "uci", output
  end
end
"""

with open("homebrew-ichess/Formula/ichess.rb", "w") as f:
    f.write(formula)

print("ichess.rb generated.")
