{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    nixpkgs_master.url = "github:NixOS/nixpkgs/master";
    nixpkgs_ank.url = "github:leoank/nixpkgs/cuda";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
  };

  outputs = { self, nixpkgs, flake-utils, systems, ... } @ inputs:
      flake-utils.lib.eachDefaultSystem (system:
        let
            pkgs = import nixpkgs {
              system = system;
              config.allowUnfree = true;
              config.cudaSupport = true;
            };

            apkgs = import inputs.nixpkgs_ank {
              system = system;
              config.allowUnfree = true;
              config.cudaSupport = true;
            };

            libList = [
                # Add needed packages here
                pkgs.stdenv.cc.cc
                pkgs.libGL
                pkgs.glib
                pkgs.zlib
              ] ++ pkgs.lib.optionals pkgs.stdenv.isLinux (with apkgs.cudaPackages_12_4; [
                # Only needed on linux env
                libcublas
                libcurand
                pkgs.cudaPackages.cudnn
                libcufft
                cuda_cudart

                # This is required for most app that uses graphics api
                pkgs.linuxPackages.nvidia_x11
              ]);

            # mpkgs = import inputs.nixpkgs_master {
            #   system = system;
            #   config.allowUnfree = true;
            # };
          in
          with pkgs;
        {
          devShells = {
              default = let
                python_with_pkgs = (pkgs.python311.withPackages(pp: [
                ]));
              in mkShell {
                    NIX_LD = runCommand "ld.so" {} ''
                        ln -s "$(cat '${pkgs.stdenv.cc}/nix-support/dynamic-linker')" $out
                      '';
                    NIX_LD_LIBRARY_PATH = lib.makeLibraryPath libList;
                    packages = [
                      python_with_pkgs
                      gcc
                    ]
                    ++ libList;
                    venvDir = "./.venv";
                    postVenvCreation = ''
                        unset SOURCE_DATE_EPOCH
                      '';
                    postShellHook = ''
                        unset SOURCE_DATE_EPOCH
                      '';
                    shellHook = ''
                        export LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
                        export PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring
                        export CUDA_HOME=${apkgs.cudaPackages_12_4.cudatoolkit}
                    '';
                  };
              };
        }
      );
}
