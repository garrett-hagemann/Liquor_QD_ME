using DataArrays, DataFrames, ForwardDiff, NLsolve, Roots, Distributions, Optim, Interpolations

#= goofy conversion magic =#
import Base.convert
function convert(t::Type{Float64},x::Array{Float64,1})
	return convert(Float64,x[1])
end

import Base.isless
function isless(x::Array{Float64,1},y::Float64)
	return isless(x[1],y)
end
function isless(x::Array{Float64,1},y::Array{Float64,1})
	return isless(x[1],y[1])
end

srand(69510606) #seeding random number gen


df = readtable("../../demand_estimation/berry_logit/berry_logit.csv")
char_list = [:price, :d_gin, :d_vod, :d_rum, :d_sch, :d_brb, :d_whs, :holiday]
coefs = readtable("../../demand_estimation/berry_logit/blogit_coefs.csv")

#= each row in df represents one bundle of p,q. Each row also contains all
the needed information to determine the cutoffs. So, we'll be looping over the relevant
price schedule segments and cutoffs and doing it for each row. =#

#= each column in coefs is the appropriate coefficient =#
coef_vec = convert(Array,coefs) # row vector
alpha = coef_vec[1,1]

function weibull_pdf(x, gamma, mu,a)
	return (gamma/a)*((x - mu)/a)^(gamma - 1)*exp(-((x - mu)/a)^gamma)
end

function unif_pdf(x)
	if 0 <= x <= 1
		return 1
	else
		return 0
	end
end

function burr_cdf(x,gamma)
	return 1 - (1-x)^(1/gamma)
end

function sparse_int(f::Function, a::Number, b::Number)
	#= Implements sparse grid quadrature from sparsegrids.de
	This implements the 1 dimensional rule that is exact for
	polynomials up to order 25.

	f = one dimensional function
	a = lower bound of integration
	b = upperbound of integration

	=#
	 weights = [.052328105232810528, .13424401342440137, .20069872006987202, .22545832254583226, .20069872006987202, .13424401342440137, .052328105232810528]

	nodes = [.01975439999999995,.11270170000000002,.28287810000000002,.5,.71712189999999998, .88729829999999998,.98024560000000005]

	f_evals = [f((b-a)*u + a) for u in nodes]
	return dot(f_evals, weights)*(b-a)
end

# defining limits of type distribution. Should be consistent with whatever PDF is used



#markets = convert(Vector
markets = [178]

for market in markets
	#products = levels(df[df[:mkt] .== market
	products = [650]
	
	for product in products
		println("Working with Market: $market, Product: $product")
		# defining selection boolians so that we can select appropriate products/markets
		prod_bool = ((df[:product] .== product) & (df[:mkt] .== market))
		other_bool = ((df[:product] .!= product) & (df[:mkt] .== market))

		prod_chars = convert(Array,df[prod_bool,char_list])
		other_chars = convert(Array,df[other_bool,char_list])

		# flag that determines if product has matched price data
		matched_upc = (df[prod_bool, :_merge_purchases][1] == 3)

		if matched_upc == true # Only need to work with matched products. No price sched for non-matched.
			tic()
			println("Product has matched price data. Evaluating cutoff prices.")
			#Defining some constants for use in the share function
			xb_prod = (prod_chars[:,2:end]*coef_vec[:,2:end]')[1] + df[prod_bool,:xi][1] # scalar
			
			if size(other_chars)[1] == 0
				xb_others = 0
			else
				xb_others = other_chars*coef_vec' + convert(Vector,df[other_bool,:xi]) # J-1 x 1
			end

			#Grabbing market size. Needs to be the same as that used to estimate shares
			M = df[prod_bool,:M][1]
			prod_price = df[prod_bool, :price][1]

			# lists of observed prices and quantities. Need these to calculate error using 
			# estimated price schedule 
			rho_list = [:actual_p0, :actual_p1, :actual_p2, :actual_p3, :actual_p4, :actual_p5, :actual_p6,
				:actual_p7, :actual_p8, :actual_p9, :actual_p10]

			cutoff_q_list = [:disc_q0, :disc_q1, :disc_q2, :disc_q3, :disc_q4, :disc_q5, :disc_q6,
				:disc_q7, :disc_q8, :disc_q9, :disc_q10]

			obs_rhos = dropna(convert(DataArray,df[prod_bool, rho_list])'[:,1]) #some goofy conversion to make the dropna work as expected
			obs_cutoff_q = dropna(convert(DataArray,df[prod_bool, cutoff_q_list])'[:,1]) #same as above
			obs_ff = [0.0] # first fixed fee is by definition 0

			
			#= calculating series of A (intercepts of piecewise linear price schedules
			The convention is that rho[k] is associated with quantities between cutoff_q[k-1]
			and cutoff_q[k] where cutoff_q[0] = 0 (i.e. rho[1] is for all quantities up to
			cutoff_q[1] and rho[2] 	is for quantities between cuttoff_q[1] and cutoff_q[2] =#
			for k in 2:length(obs_rhos) #mismatch between the indexing. take care
				y_intersect = obs_rhos[k-1]*obs_cutoff_q[k] # what would price be if you bought at previous unit price?
				intercept = y_intersect - obs_rhos[k]*obs_cutoff_q[k] # where does current unit price interesct vertical axis?
				push!(obs_ff,intercept) # append to list obs_ff
			end


			#Defining share function which is ONLY a function of price
			function share(p)
				#= Function calculates shares of a given  product in a given market 
				based on price. Characteristics of the product and all other products
				are assumed fixed. 

				p = price =#
				
				num = exp(alpha*p + xb_prod)
				denom = 1 + num + sum(exp(xb_others))
				
				s = num/denom	

				return s[1] # dealing with weird type promotion stuff
			end

			#Derivatives of the share fuction
			d_share(p) = ForwardDiff.derivative(share, p[1]) #p should be scalar. Optimization routine returns a 1 element array at some point
			dd_share(p) = ForwardDiff.derivative(d_share,p[1])
			ddd_share(p) = ForwardDiff.derivative(dd_share,p[1])

			function p_star(rho,l)
				function g!(p,gvec)
					gvec[1] =  (p - rho - l)*d_share(p) + share(p)
				end
				function gj!(p, gjvec)
					gjvec[1] = (p - rho - l)*dd_share(p) + 2.0*d_share(p)
				end
				
				res = nlsolve(g!,gj!,[rho+l],show_trace = false, method = :trust_region, extended_trace = false) 
				return res.zero[1]
			end
			#=Interpolationg p_star at a bunch of points to help speed up calculation =#
			p_star_grid = [p_star(i,j) for i = 1.0:100.0, j = 1.0:100.0]
			p_star_grid = convert(Array{Float64,2},p_star_grid)
			p_star_itp = interpolate(p_star_grid, BSpline(Quadratic(Reflect())), OnGrid())
			function d_pstar_d_rho(rho,lambda) # note that this is the same as d_pstar_d_lambda
				u = p_star(rho,lambda)
				res = d_share(u) / (dd_share(u)*(u - rho - lambda) + 2*d_share(u))
				return res[1]
			end
			function d_pstar_d_lambda(rho,lambda) # see above
				return d_pstar_d_rho(rho,lambda)
			end
			function d2_pstar_d2_rho(rho,lambda)
				u = p_star(rho,lambda)
				num1 = (dd_share(u)*(u - rho - lambda) + 2*d_share(u))*dd_share(u)*d_pstar_d_rho(rho,lambda)
				num2 = d_share(u)*(dd_share(u)*(d_pstar_d_rho(rho,lambda) - 1) + ((u - rho - lambda)*ddd_share(u) + 2*dd_share(u))*d_pstar_d_rho(rho,lambda))
				den = dd_share(u)*(u - rho - lambda) + 2*d_share(u)
				res = (num1 - num2)/(den^2)
D
				return res
			end

			function price_sched_calc(params,N)
				# params are the params of the wholesaler's problem we're trying to
				# estimate. 

				# Function returns the coefficients defining the price schedule as [rho, lambda]
				# where rho is n long and lambda is n-1 long for an n option schedule

				c = params[1] # marginal cost for wholesaler
				max_mc = params[2] # max MC for retailer. Scales type parameter
				a = exp(params[3]) # first param for Beta distribution
				b = exp(params[4]) # second param for Beta distribution
				#M = exp(params[5])
				lambda_lb = 0
				lambda_ub = max_mc
				est_pdf(x) = pdf(Beta(a,b),x/max_mc)
				est_cdf(x) = cdf(Beta(a,b),x/max_mc)

				#= finding rho_0. rho_0 is such that the lowest type is indifferent about purchasing (entering the market)
				For retailers, this equates to a 0-profit condition:
				(p_star(rho_0,lambda_ub) - rho_0 - lambda_ub)*share(p_star(rho_0,lambda_ub)) = 0

				function low_profit!(rho_0, lpvec)
					u = p_star(rho_0,lambda_lb)
					lpvec[1] = (u - rho_0 - lambda_lb)*share(u)
				end
				rho_0_sol = nlsolve(low_profit!,[12.0])
				rho_0 = rho_0_sol.zero[1]
				=#
				rho_0 = 1000
				
				##### CODE THAT USES NLSOLVE #####

				# Defining Wholesaler FOCs which can then be solved. Following the syntax to use
				# NLsovle package.
				# NOTE: Can refomulate as constrained minimization if this doesn't work

				function wfocs!(theta::Vector, wfocs_vec)
					K = N-1 # Number of rhos / lambdas
									
					lambda_vec = [lambda_ub; theta[K+1:end]; lambda_lb] # not that these are reversed: i.e. for N = 4 -> [max_mc, lambda_1, lambda_2, lambda_3, lambda_4 = lambda_lb = 0]. THis is because high lambdas are bad such that lambda_k > lambda_k+1
D
					rho_vec = [rho_0 ; theta[1:K]] # that inital value is just to force share(p_star(rho_0,lambda_0)) = 0. Ideally should be Inf
					# Calculating FOCs
					for i in 2:K+1 # need to up index by 1 to reference right coefs since vectors start indexing at 1
						#calculating integral for rho FOC
						f(l) =  ((rho_vec[i] - c)*d_share(p_star(rho_vec[i],l))*d_pstar_d_rho(rho_vec[i],l) + share(p_star(rho_vec[i],l)))*est_pdf(l)
						int = sparse_int(f,lambda_vec[i+1], lambda_vec[i])
						# Pre-calculating some stuff to avoid repeated calls to p_star
						ps1 = p_star(rho_vec[i],lambda_vec[i])
						ps2 = p_star(rho_vec[i-1],lambda_vec[i])
						#rho FOC for k
						wfocs_vec[i-1] = int + est_cdf(lambda_vec[i])*((ps1 - rho_vec[i] - lambda_vec[i])*d_share(ps1)*d_pstar_d_rho(rho_vec[i],lambda_vec[i]) + share(ps1)*(d_pstar_d_rho(rho_vec[i],lambda_vec[i]) - 1)) 
						# lambda FOC for k
						wfocs_vec[i+K-1] = (rho_vec[i] - c)*share(ps1)*est_pdf(lambda_vec[i]) + est_cdf(lambda_vec[i])*((ps1 - rho_vec[i] - lambda_vec[i])*d_share(ps1)*d_pstar_d_lambda(rho_vec[i],lambda_vec[i]) + share(ps1)*(d_pstar_d_lambda(rho_vec[i],lambda_vec[i]) - 1) - (ps2 - rho_vec[i-1] - lambda_vec[i])*d_share(ps2)*d_pstar_d_lambda(rho_vec[i-1],lambda_vec[i]) - share(ps2)*(d_pstar_d_lambda(rho_vec[i-1],lambda_vec[i]) - 1)) + ((ps1 - rho_vec[i] - lambda_vec[i])*share(ps1) - (ps2 - rho_vec[i-1] - lambda_vec[i])*share(ps2))*est_pdf(lambda_vec[i])
						
					end
					print(".") # print . for each eval of wfocs
				end
				
				srand(1987) #reseeding to x0 is same every time
				x0 = ones(2*N-2) + randn(2*N-2)
				solution = nlsolve(wfocs!,x0,method=:trust_region, show_trace = false, ftol = 1e-8, extended_trace = false, iterations = 1000)
				#solution = nlsolve(wfocs!,wfocs_jac!,x0,method=:trust_region, show_trace = false, ftol = 1e-8, extended_trace = false, iterations = 1000)
				est_rhos = [rho_0; solution.zero[1:N-1]] # need to return this. These are the prices for the price schedule
				est_lambdas = [lambda_ub; solution.zero[N:end]; lambda_lb] # remaining parameters are lambda cutoffs
				
				# Calculating fixed fees
				est_ff = [0.0]
				K = N-1
				for i in 2:K
					A = est_ff[i-1] + (p_star(est_rhos[i],est_lambdas[i]) - est_rhos[i] - est_lambdas[i])*share(p_star(est_rhos[i],est_lambdas[i]))*M - (p_star(est_rhos[i-1],est_lambdas[i]) - est_rhos[i-1] - est_lambdas[i])*share(p_star(est_rhos[i-1],est_lambdas[i]))*M
					push!(est_ff,A)
				end
				return (est_rhos,est_ff,est_lambdas)
			end
			
			function obj_func(omega::Vector, N::Int, W::Matrix)
				tic()
				rho_hat,ff_hat,lambda_hat = price_sched_calc(omega,N)
				vec = [(rho_hat[2:end] - obs_rhos) ; (ff_hat[2:end] - obs_ff[2:end])]'
				res = vec*W*vec'
				toc()
				return res[1]
			end
			N = length(obs_rhos)+1
			W = eye(2*(N-1)- 1)
			#W = Diagonal([1./(obs_rhos.^2) ; 1./(obs_ff[2:end].^2)])*eye(2*N-2)
			x0 = [0.0, 40.0, 0.0, 0.0]
			g(x) = obj_func(x,N,W)
			optim_res = optimize(g,x0,NelderMead(),OptimizationOptions(show_every = true, iterations = 3000))
			println(optim_res)
			min_X = Optim.minimizer(optim_res)
			println(min_X)
			println(price_sched_calc(min_X,N))
			println(obs_rhos)
			println(obs_ff)
			toc()
		else
			println("Product has no matching price data.")
		end
	end
end
