module Moreau_CUDA_jll

using Artifacts
using Downloads
using Libdl
using SHA

const libmoreau_cuda_path = Ref{String}("")
const libmoreau_cuda_handle = Ref{Ptr{Cvoid}}(C_NULL)

# libmoreau_cuda is the symbol used in ccall — it must resolve to a library path.
# Empty string means CUDA library is not available.
libmoreau_cuda::String = ""

# Whether the CUDA library was successfully loaded
const _available = Ref{Bool}(false)

function __init__()
    # Priority 1: explicit env var (local development)
    path = get(ENV, "MOREAU_CUDA_LIB", "")

    # Priority 2: Julia Artifacts system (authenticated download from Gemfury)
    if isempty(path)
        artifacts_toml = joinpath(pkgdir(Moreau_CUDA_jll), "Artifacts.toml")
        if isfile(artifacts_toml)
            for artifact_name in _cuda_artifact_order()
                hash = artifact_hash(artifact_name, artifacts_toml;
                    platform=Base.BinaryPlatforms.HostPlatform())
                hash === nothing && continue
                if !artifact_exists(hash)
                    _download_from_gemfury(artifacts_toml, artifact_name, hash)
                end
                if artifact_exists(hash)
                    path = _find_lib_in_artifact(artifact_path(hash))
                    !isempty(path) && break
                end
            end
        end
    end

    # Priority 3: system library search
    if isempty(path)
        path = something(
            Libdl.find_library("moreau_cuda"),
            Libdl.find_library("libmoreau_cuda"),
            "",
        )
    end

    if isempty(path)
        @debug "Moreau CUDA library not found — CUDA backend unavailable. " *
               "Set MOREAU_CUDA_LIB or install the CUDA artifact."
        global libmoreau_cuda = ""
        _available[] = false
        return
    end

    try
        global libmoreau_cuda = path
        libmoreau_cuda_path[] = path
        libmoreau_cuda_handle[] = Libdl.dlopen(path)
        _available[] = true
    catch e
        @warn "Failed to load Moreau CUDA library" path exception=(e, catch_backtrace())
        global libmoreau_cuda = ""
        _available[] = false
    end
end

is_available() = _available[]

"""
    _cuda_artifact_order() -> Tuple{String, ...}

Return artifact names to try, in priority order:
1. `MOREAU_CUDA_VERSION` env var (e.g. "12" or "13") — use only that version
2. `nvidia-smi` — detect max supported CUDA version, prefer matching major
3. Fallback — try cuda13 first, then cuda12
"""
function _cuda_artifact_order()
    # Priority 1: explicit env var
    env_ver = get(ENV, "MOREAU_CUDA_VERSION", "")
    if !isempty(env_ver)
        major = _parse_cuda_major(env_ver)
        if major !== nothing
            @debug "MOREAU_CUDA_VERSION=$env_ver — using cuda$major"
            return ("moreau_cuda$major",)
        else
            @warn "MOREAU_CUDA_VERSION=$env_ver is not a valid CUDA version, ignoring"
        end
    end

    # Priority 2: detect from nvidia-smi
    detected = _detect_cuda_major()
    if detected !== nothing
        @debug "nvidia-smi reports CUDA $detected"
        if detected >= 13
            return ("moreau_cuda13", "moreau_cuda12")
        else
            return ("moreau_cuda12",)
        end
    end

    # Priority 3: fallback — prefer newer
    return ("moreau_cuda13", "moreau_cuda12")
end

function _parse_cuda_major(s::AbstractString)::Union{Int, Nothing}
    # Accept "12", "13", "12.2", "13.0", etc.
    m = match(r"^(\d+)", strip(s))
    m === nothing && return nothing
    major = parse(Int, m.captures[1])
    return major in (12, 13) ? major : nothing
end

function _detect_cuda_major()::Union{Int, Nothing}
    try
        output = read(`nvidia-smi`, String)
        # nvidia-smi header contains e.g. "CUDA Version: 13.0"
        m = match(r"CUDA Version:\s*(\d+)", output)
        m === nothing && return nothing
        return parse(Int, m.captures[1])
    catch
        return nothing
    end
end

function _find_lib_in_artifact(artifact_dir::String)
    lib_dir = joinpath(artifact_dir, "lib")
    if Sys.isapple()
        candidate = joinpath(lib_dir, "libmoreau_cuda.dylib")
    else
        candidate = joinpath(lib_dir, "libmoreau_cuda.so")
    end
    return isfile(candidate) ? candidate : ""
end

function _download_from_gemfury(artifacts_toml::String, artifact_name::String, hash::Base.SHA1)
    token = get(ENV, "GEMFURY_TOKEN", "")
    if isempty(token)
        @debug "GEMFURY_TOKEN not set — cannot download Moreau CUDA library"
        return
    end

    meta = artifact_meta(artifact_name, artifacts_toml;
        platform=Base.BinaryPlatforms.HostPlatform())
    if meta === nothing
        @debug "No $(artifact_name) artifact entry for this platform"
        return
    end

    wheel_filename = get(meta, "wheel_filename", nothing)
    wheel_sha256 = get(meta, "wheel_sha256", nothing)

    if wheel_filename === nothing
        @debug "Artifacts.toml missing wheel_filename for $(artifact_name)"
        return
    end

    # Derive PyPI package name from artifact name: moreau_cuda12 -> moreau-clib-cuda12
    pypi_pkg = replace(artifact_name, "moreau_" => "moreau-clib-")

    url = _find_wheel_on_gemfury(token, pypi_pkg, wheel_filename)
    if url === nothing
        @warn "Could not find $(wheel_filename) on Gemfury"
        return
    end

    # Derive wheel internal directory from package name: moreau-clib-cuda12 -> moreau_clib_cuda12
    pkg_dir = replace(pypi_pkg, "-" => "_")

    local_wheel = joinpath(tempdir(), wheel_filename)
    try
        @info "Downloading Moreau CUDA library ($(artifact_name))..."
        Downloads.download(url, local_wheel)

        if wheel_sha256 !== nothing
            actual = bytes2hex(open(sha256, local_wheel))
            if actual != wheel_sha256
                error("SHA256 mismatch for $(wheel_filename): expected $(wheel_sha256), got $(actual)")
            end
        end

        _install_wheel_as_artifact(local_wheel, hash, pkg_dir, "libmoreau_cuda")
    catch e
        @warn "Failed to download Moreau CUDA library" exception=(e, catch_backtrace())
    finally
        rm(local_wheel; force=true)
    end
end

"""
    _find_wheel_on_gemfury(token, package, filename) -> Union{String, Nothing}

Query the Gemfury PyPI simple index for `package` and return the authenticated
download URL for `filename`, or `nothing` if not found.
"""
function _find_wheel_on_gemfury(token::String, package::String, filename::String)
    index_url = "https://$(token)@pypi.fury.io/optimalintellect/$(package)/"
    try
        html = String(take!(Downloads.download(index_url, IOBuffer())))
        for m in eachmatch(r"href=\"([^\"]+)\"", html)
            href = m.captures[1]
            if endswith(href, filename) || contains(href, "/$(filename)")
                if startswith(href, "http")
                    if !contains(href, "@")
                        href = replace(href, "https://" => "https://$(token)@")
                    end
                    return href
                else
                    return "https://$(token)@pypi.fury.io/optimalintellect/$(package)/$(filename)"
                end
            end
        end
    catch e
        @warn "Failed to query Gemfury PyPI index" exception=(e, catch_backtrace())
    end
    return nothing
end

"""
    _install_wheel_as_artifact(wheel, expected_hash, pkg_dir, lib_stem)

Extract a shared library from a wheel (zip) into a Julia artifact.
"""
function _install_wheel_as_artifact(wheel::String, expected_hash::Base.SHA1,
                                     pkg_dir::String, lib_stem::String)
    hash = create_artifact() do dir
        lib_dir = joinpath(dir, "lib")
        mkpath(lib_dir)
        run(`unzip -j -o $wheel "$pkg_dir/$lib_stem.*" -d $lib_dir`)
    end
    if hash != expected_hash
        remove_artifact(hash)
        error(
            "Artifact tree hash mismatch: expected $(bytes2hex(expected_hash.bytes)), " *
            "got $(bytes2hex(hash.bytes))"
        )
    end
end

export libmoreau_cuda, is_available

end # module
