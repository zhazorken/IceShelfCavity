
""" Centered boxcar smoothing of a series y(x) over a window `width` (same units as x).
    Used to smooth the ice-base topography before building the immersed boundary. """
function smooth_series(y, x, width)
    width <= 0 && return collect(float.(y))
    Δ = abs(x[2] - x[1])
    half = max(1, round(Int, (width / Δ) / 2))
    n = length(y)
    ys = zeros(float(eltype(y)), n)
    for i in 1:n
        lo = max(1, i - half); hi = min(n, i + half)
        ys[i] = sum(@view y[lo:hi]) / (hi - lo + 1)
    end
    return ys
end

""" Rounds `a` to the nearest even number """
nearest_even(a) = Int(2*round(round(a) / 2))

""" Returns Nx Ny Nz based on total number of points `N`"""
function get_sizes(N; Lx=8, Ly=8, Lz=0.5, aspect_ratio_x=3.2, aspect_ratio_y=3.2)
    Cz = Lx * Ly / (Lz^2 * aspect_ratio_x * aspect_ratio_y)
    Nz = (N / Cz)^(1/3)
    Nx = Lx * Nz / (Lz * aspect_ratio_x)
    Ny = Ly * Nz / (Lz * aspect_ratio_y)
    Nx = round(Int, Nx)
    Ny = round(Int, Ny)
    Nz = round(Int, max(2, Nz))
    return (; Nx, Ny, Nz)
end

# Function to generate multiples of prime factors up to a limit
function closest_factor_number(primes::NTuple{3, Int}, target::Int)
    closest_number = 1
    min_difference = abs(target - closest_number)
    # We will iterate over different combinations of powers of primes
    for i in 0:15  # You can adjust this loop depth
        for j in 0:15
            for k in 0:15
                # Generate the product of primes with different powers
                product = primes[1]^i * primes[2]^j * primes[3]^k
                diff = abs(target - product)
                if diff < min_difference
                    min_difference = diff
                    closest_number = product
                end
            end
        end
    end
    return closest_number
end

function closest_factor_number(primes::NTuple{2, Int}, target::Int)
    closest_number = 1
    min_difference = abs(target - closest_number)
    # We will iterate over different combinations of powers of primes
    for i in 0:15 # You can adjust this loop depth
        for j in 0:15
            # Generate the product of primes with different powers
            product = primes[1]^i * primes[2]^j
            diff = abs(target - product)
            if diff < min_difference
                min_difference = diff
                closest_number = product
            end
        end
    end
    return closest_number
end

function closest_factor_number(primes::NTuple{1, Int}, target::Int)
    closest_number = 1
    min_difference = abs(target - closest_number)
    # We will iterate over different combinations of powers of primes
    for i in 0:15 # You can adjust this loop depth
        # Generate the product of primes with different powers
        product = primes[1]^i
        diff = abs(target - product)
        if diff < min_difference
            min_difference = diff
            closest_number = product
        end
    end
    return closest_number
end
