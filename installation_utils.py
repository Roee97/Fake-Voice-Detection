import subprocess
import sys


def install_dependencies():
    """Installs necessary Python packages for the project."""
    packages = [
        "datasets",
        "git+https://github.com/One-sixth/fairseq.git",
        "wavencoder",
        "omegaconf==2.1.2"
    ]

    # Uninstall omegaconf before reinstalling it
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "omegaconf", "--yes"], check=True)

    # Install each package
    for package in packages:
        subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
