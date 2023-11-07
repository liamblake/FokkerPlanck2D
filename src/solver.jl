using SparseArrays

using OrdinaryDiffEq


"""

Numerically solves the Fokker-Planck equation
    ρₜ = -∇·(ρ u) + ∇·(D ∇ρ)
corresponding to the PDF of the solution to the SDE
    dx = u(x,t)dt + σ(x,t)dWₜ,
where D = σσᵀ is diagonal.

"""
function _solve_2d_fp!(dest, u!, ∇u!, D!, ∇D!, ∇∇D!, xgrid, ygrid, ρ₀, tspan; ode_solver=Tsit5(), solver_kwargs...)
    # Setup as a differential algebraic equation with a mass matrix, enforcing zero density at the
    # spatial boundaries.

    # All pairs of grid points
    nx = length(xgrid)
    ny = length(ygrid)
    grid = [[x, y] for x in xgrid[2:(end-1)], y in ygrid[2:(end-1)]][:]
    dx = diff(xgrid)
    dy = diff(ygrid)

    # Mass matrix
    M = spdiagm(0 => ones(nx * ny))
    M[1:nx:end, 1:nx:end] = 0.0
    M[1:nx, 1:nx] = 0.0
    M[(ny-1)*nx:end, (ny-1)*nx:end] = 0.0
    M[nx:nx:end, nx:nx:end] = 0.0

    # Initial state
    u0 = ρ₀.(grid)

    # Temporary storage of states - to reduce allocations
    vtmp = Vector{Float64}(undef, 2)
    vtmp2 = Vector{Float64}(undef, 2)
    mtmp = Matrix{Float64}(undef, 2, 2)

    # RHS of equation
    function fp_rhs!(dp, p, _, t)
        # Boundaries
        dp[1:nx:end] = p[1:nx:end]
        dp[1:nx] = p[1:nx]
        dp[(ny-1)*nx:end] = p[(ny-1)*nx:end]
        dp[nx:nx:end] = p[nx:nx:end]

        # Interior
        for (i, x) in enumerate(xgrid[2:(end-1)])
            for (j, y) in enumerate(ygrid[2:(end-1)])

                ∇u!(mtmp, x, y, t)

                # Contribution from current cell
                ∇∇D!(vtmp, x, y, t)
                dp[i+1, j+1] = (-mtmp[1, 1] - mtmp[2, 2] + 0.5 * vtmp[1] + 0.5 * vtmp[2]) * p[i+1, j+1]

                D!(mtmp, x, y, t)
                dp[i+1, j+1] -= (1 / dx[i]^2 * mtmp[1, 1] + 1 / dy[j]^2 * mtmp[2, 2]) * p[i+1, j+1]

                # Contribution from east cell
                ∇D!(vtmp2, x, y, t)
                u!(vtmp, x, y, t)
                dp[i+1, j+1] += (
                    -1 / (2 * dx[i]) * vtmp[1] + 1 / (2 * dx[i]^2) * mtmp[1, 1] + 1 / (2 * dx[i]) * vtmp2[1]
                ) * p[i+2, j+1]

                # Contribution from west cell
                dp[i+1, j+1] += (
                    1 / (2 * dx[i]) * vtmp[1] + 1 / (2 * dx[i]^2) * mtmp[1, 1] - 1 / (2 * dx[i]) * vtmp2[1]
                ) * p[i, j+1]

                # Contribution from north cell
                dp[i+1, j+1] += (
                    -1 / (2 * dy[j]) * vtmp[2] + 1 / (2 * dy[j]^2) * mtmp[2, 2] + 1 / (2 * dy[j]) * vtmp2[2]
                ) * p[i+1, j+2]

                # Contribution from south cell
                dp[i+1, j+1] += (
                    1 / (2 * dy[j]) * vtmp[2] + 1 / (2 * dy[j]^2) * mtmp[2, 2] - 1 / (2 * dy[j]) * vtmp2[2]
                ) * p[i+1, j]

            end
        end
    end

    # Solve the problem
    rhs = ODEFunction(fp_rhs!, mass_matrix=M)
    prob = ODEProblem(rhs, u0, tspan)
    sol = solve(prob, ode_solver; save_everystep=false, save_start=false, solver_kwargs...)

    dest .= reshape(Array(sol.u), nx, ny)

end