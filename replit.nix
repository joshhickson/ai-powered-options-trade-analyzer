
{ pkgs }: {
  deps = [
    pkgs.python3
    pkgs.python3Packages.pip
    pkgs.python3Packages.pandas
    pkgs.python3Packages.numpy
    pkgs.python3Packages.scipy
    pkgs.python3Packages.requests
    pkgs.python3Packages.yfinance
    pkgs.python3Packages.matplotlib
    pkgs.python3Packages.urllib3
    pkgs.python3Packages.setuptools
  ];
  env = {
    PYTHONPATH = "${pkgs.python3Packages.pip}/lib/python3.11/site-packages";
  };
}
