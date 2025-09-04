import sys

if len(sys.argv) != 2:
    print("Usage: python3 darknetm.py <sha256>")
    sys.exit(1)

sha256 = sys.argv[1]

formula = f"""class Ichess < Formula
  desc "Darknet on OpenCL CNN Engine with GPU acceleration"
  homepage "https://isowa.io"
  url "https://github.com/sowson/darknet/releases/download/darknet-v{{{{version}}}}/Darknet.mac.zip"
  sha256 "{sha256}"
  license "MIT"
  version ENV["ICHESS_VERSION"] || "1.0"

  depends_on "clblas"
  depends_on "opencv"

  def install
    bin.install "Darknet.mac" => "darknet"
  end

  def caveats
    <<~EOS
      The Darknet engine has been installed as `darknet`.
    EOS
  end
end
"""

with open("homebrew-darknet/Formula/darknet.rb", "w") as f:
    f.write(formula)

print("darknet.rb generated.")
