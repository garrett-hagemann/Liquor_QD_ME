using DataArrays, DataFrames, ForwardDiff, NLsolve, Roots, Distributions, Optim, Interpolations, LineSearches

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

@everywhere function sparse_int(f::Function, a::Number, b::Number)
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
	return dot(f_evals, weights)*(b-a)::Float64
end

markets = convert(Vector, levels(df[:,:mkt]))
#markets = [178]

@everywhere csvfile = open("indirect_est.csv", "w")
write(csvfile, "product,mkt,c,lambda_ub,a,b,price_sched\n")
flush(csvfile)

@sync @parallel for market in markets
	products = levels(df[df[:mkt] .== market, :product])
	#products = [650]
	
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
			println("Product has matched price data.")
			#Defining some constants for use in the share function
			xb_prod = (prod_chars[:,2:end]*coef_vec[:,2:end]')[1] + df[prod_bool,:xi][1] # scalar
			
			if size(other_chars)[1] == 0
				xb_others = 0
			else
				xb_others = other_chars*coef_vec' + convert(Vector,df[other_bool,:xi]) # J-1 x 1
			end

			#Grabbing market size. Needs to be the same as that used to estimate shares
			#M = df[prod_bool,:M][1]
			M = 1.0
			prod_price = df[prod_bool, :price][1]

			# lists of observed prices and quantities. Need these to calculate error using 
			# estimated price schedule 
			rho_list = [:actual_p0, :actual_p1, :actual_p2, :actual_p3, :actual_p4, :actual_p5, :actual_p6,
				:actual_p7, :actual_p8, :actual_p9, :actual_p10]

			cutoff_q_list = [:disc_q0, :disc_q1, :disc_q2, :disc_q3, :disc_q4, :disc_q5, :disc_q6,
				:disc_q7, :disc_q8, :disc_q9, :disc_q10]

			obs_rhos = dropna(convert(DataArray,df[prod_bool, rho_list])'[:,1]) #some goofy conversion to make the dropna work as expected
			obs_cutoff_q = dropna(convert(DataArray,df[prod_bool, cutoff_q_list])'[:,1]) #same as above
			obs_ff = [0.0] # first fixed fee is by definition
			
			#max_mc = copy(maximum(obs_rhos)*1.50) # 50% buffer
			max_mc = 15.0
			# calculating series of A (intercepts of piecewise linear price schedules
			# calculating tariff
			for k = 2:length(obs_cutoff_q)
				diff = (obs_cutoff_q[k] - 1)*(obs_rhos[k-1] - obs_rhos[k])
				res = diff + obs_ff[k-1]
				push!(obs_ff,res)
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

			### NLsolve pstar
			#=	
			function p_star(rho,l)
				function g!(p,gvec)
					gvec[1] =  (p - rho + l)*d_share(p)*M + share(p)*M
				end
				function gj!(p, gjvec)
					gjvec[1] = (p - rho + l)*dd_share(p)*M + 2.0*d_share(p)*M
				end
				res = nlsolve(g!,gj!,[rho-l],show_trace = false, extended_trace = false, method = :trust_region) 
				return res.zero[1]
			end
			=#	
			### Optimizer pstar
				
			function p_star(rho::Float64,l::Float64)
				function g(p::Float64) # retailer profit
					return -((p - rho + l)*share(p)*M)[1] # Ignores fixed cost as it just shifts problem
				end
				function gj!(p,gvec)
					gvec[1] =  -((p - rho + l)*d_share(p)*M + share(p)*M)
				end
				function gh!(p, gjvec)
					gjvec[1] = -((p - rho + l)*dd_share(p)*M + 2.0*d_share(p)*M)
				end
				poptim_res = optimize(g,0.0,100.0, method=Brent(), show_trace = false, extended_trace = false, rel_tol = 1e-6, abs_tol = 1e-6)
				#poptim_res = optimize(uLg,focs!,ux0,BFGS(),OptimizationOptions(show_every = true, extended_trace = true, iterations = 1500, g_tol = 1e-6))
				return Optim.minimizer(poptim_res)[1]
			end

			### Roots pstar
			#=
			function p_star(rho::Float64,l::Float64)
				function g(p)
					return (p - rho + l)*d_share(p) + share(p)
				end
				res = fzero(g,(rho-l))
				return res
			end
			=#
			function d_pstar_d_rho(rho::Float64,lambda::Float64) # note that this is the same as d_pstar_d_lambda
				u = p_star(rho,lambda)
				res = d_share(u) ./ (dd_share(u)*(u - rho + lambda) + 2*d_share(u))
				return res[1]
			end
			function d_pstar_d_lambda(rho::Float64,lambda::Float64) # see above
				return -d_pstar_d_rho(rho,lambda)
			end
			function d2_pstar_d2_rho(rho::Float64,lambda::Float64)
				u = p_star(rho,lambda)
				num1 = (dd_share(u)*(u - rho + lambda) + 2*d_share(u))*dd_share(u)*d_pstar_d_rho(rho,lambda)
				num2 = d_share(u)*(dd_share(u)*(d_pstar_d_rho(rho,lambda) - 1) + ((u - rho + lambda)*ddd_share(u) + 2*dd_share(u))*d_pstar_d_rho(rho,lambda))
				den = dd_share(u)*(u - rho + lambda) + 2*d_share(u)
				res = (num1 - num2)/(den^2)
				return res
			end
			function d2_pstar_d2_lambda(rho::Float64,lambda::Float64)
				u = p_star(rho,lambda)
				num1 = (dd_share(u)*(u - rho + lambda) + 2*d_share(u))*dd_share(u)*d_pstar_d_lambda(rho,lambda)
				num2 = d_share(u)*(dd_share(u)*(d_pstar_d_lambda(rho,lambda) + 1) + ((u - rho + lambda)*ddd_share(u) + 2*dd_share(u))*d_pstar_d_lambda(rho,lambda))
				den = dd_share(u)*(u - rho + lambda) + 2*d_share(u)
				res = (-num1 + num2)/(den^2)
				return res
			end
			function d2_pstar_d_rho_d_lambda(rho::Float64,lambda::Float64) # cross derivative
				return -d2_pstar_d2_lambda(rho,lambda)
			end
			function d2_pstar_d_lambda_d_rho(rho::Float64,lambda::Float64)
				return -d2_pstar_d2_rho(rho,lambda)
			end
			# Defining parameters for linear approx to demand to give hot start to non-linear version.
			# apporximating demand around observed product price
			LA = share(prod_price) - d_share(prod_price)*prod_price
			LB = d_share(prod_price)
			function Lshare(p::Float64)
				return LA + LB*p
			end
			Ld_share(p) = LB
			Ldd_share(p) = 0.0
			Lddd_share(p) = 0.0

			function Lp_star(rho, l)
				return -LA/(2.0*LB) + (rho - l)/2.0
			end
			function Ld_pstar_d_rho(rho,lambda)
				return 0.5
			end
			function Ld_pstar_d_lambda(rho,lambda)
				return -0.5
			end
			function Ld2_pstar_d2_rho(rho,lambda)
				return 0.0
			end
			function Ld2_pstar_d2_lambda(rho,lambda)
				return 0.0
			end
			function Ld2_pstar_d_rho_d_lambda(rho,lambda)
				return 0.0
			end
			function Ld2_pstar_d_lambda_d_rho(rho,lambda)
				return 0.0
			end

			function Lprice_sched_calc(params::Array{Float64,1},N::Int)
				# params are the params of the wholesaler's problem we're trying to
				# estimate. 

				# Function returns the coefficients defining the price schedule as [rho, lambda]
				# where rho is n long and lambda is n-1 long for an n option schedule

				c = params[1] # marginal cost for wholesaler
				#max_mc = params[2] # max MC for retailer. Scales type parameter
				#M = exp(params[5])
				#distribution parameters
				a = exp(params[2]) #
				b = exp(params[3]) # 
				lambda_lb = 0 #-max_mc
				lambda_ub = max_mc
					
				est_cdf(x) = cdf(Beta(a,b),(x-lambda_lb)/(lambda_ub - lambda_lb))
				est_pdf(x) = pdf(Beta(a,b),(x-lambda_lb)/(lambda_ub - lambda_lb))/(lambda_ub - lambda_lb)
				d_est_pdf(x) = ((1/(lambda_ub - lambda_lb))^2)*((a + b - 2)*(x-lambda_lb)/(lambda_ub - lambda_lb) - (a - 1))/(((x-lambda_lb)/(lambda_ub - lambda_lb) -1)*(x-lambda_lb)/(lambda_ub - lambda_lb))*pdf(Beta(a,b),(x-lambda_lb)/(lambda_ub - lambda_lb))
				
				#=
				#= below formulation of th Kumaraswarmy distribution isn't exactly right. Should only be defined on (0,1) and pdf is 0 outside that. Programming that in makes it act funny =#		
				est_cdf(x) = 1.0 - (1.0-(x/max_mc)^a)^b
				est_pdf(x) = ForwardDiff.derivative(est_cdf,x)
				d_est_pdf(x) = ForwardDiff.derivative(est_pdf,x)
				=#
				
				function urho(k::Integer)
					return -LA/LB + lambda_ub + 2*(c + LA/LB - lambda_ub)/(3-4*N)*(1 - N - k)
				end
				function ulambda(k::Integer)
					return lambda_ub - (2*(c + LA/LB - lambda_ub))/(3-4*N)*(N - k)
				end
				rho_0 = urho(0)
				#rho_0 = -A/B
				##### Optimizer Approach
				function Lw_profit(theta)
					lambda_vec = [lambda_lb; theta[N:end]; lambda_ub] 
					rho_vec = [rho_0 ; theta[1:N-1]]
					profit = 0.0
					for i = 1:N-1
						k = i+1 # dealing with indexing
						f(l) =  ((rho_vec[k] - c)*Lshare(Lp_star(rho_vec[k],l))*M)*est_pdf(l)
						int = sparse_int(f,lambda_vec[k],lambda_vec[k+1])
						# Pre-calculating some stuff to avoid repeated calls to p_star
						ps1 = Lp_star(rho_vec[k],lambda_vec[k])
						ps2 = Lp_star(rho_vec[k-1],lambda_vec[k])
						inc = int + (1-est_cdf(lambda_vec[k]))*((ps1 - rho_vec[k] + lambda_vec[k])*Lshare(ps1)*M - (ps2 - rho_vec[k-1] + lambda_vec[k])*Lshare(ps2)*M)
						profit = profit + inc
					end
					return -profit
				end
				function Lwfocs!(theta::Vector, wfocs_vec)
					lambda_vec = [lambda_lb; theta[N:end]; lambda_ub] 
					rho_vec = [rho_0 ; theta[1:N-1]] # that inital value is just to force share(p_star(rho_0,lambda_0)) = 0. Ideally should be Inf
					# Calculating FOCs
					for i in 1:N-1 
						k = i+1 # dealing with the increment in indexing and making code look more like notation
						#calculating integral for rho FOC
						f(l) =  ((rho_vec[k] - c)*Ld_share(Lp_star(rho_vec[k],l))*M*Ld_pstar_d_rho(rho_vec[k],l) + Lshare(Lp_star(rho_vec[k],l))*M)*est_pdf(l)
						int = sparse_int(f,lambda_vec[k],lambda_vec[k+1])
						# Pre-calculating some stuff to avoid repeated calls to p_star
						ps1 = Lp_star(rho_vec[k],lambda_vec[k])
						ps2 = Lp_star(rho_vec[k-1],lambda_vec[k])
						ps3 = Lp_star(rho_vec[k],lambda_vec[k+1])
						#rho FOC
						if i == N-1
							term1 = (ps1 - rho_vec[k] + lambda_vec[k])*Ld_share(ps1)*M*Ld_pstar_d_rho(rho_vec[k],lambda_vec[k]) + Lshare(ps1)*M*(Ld_pstar_d_rho(rho_vec[k],lambda_vec[k]) - 1)
							
							res = int + (1-est_cdf(lambda_vec[k]))*term1
							wfocs_vec[i] = -res
						else
							term1 = (ps1 - rho_vec[k] + lambda_vec[k])*Ld_share(ps1)*M*Ld_pstar_d_rho(rho_vec[k],lambda_vec[k]) + Lshare(ps1)*M*(Ld_pstar_d_rho(rho_vec[k],lambda_vec[k]) - 1)
							term2 = (ps3 - rho_vec[k] + lambda_vec[k+1])*Ld_share(ps3)*M*Ld_pstar_d_rho(rho_vec[k],lambda_vec[k+1]) + Lshare(ps3)*M*(Ld_pstar_d_rho(rho_vec[k],lambda_vec[k+1]) - 1)
							
							res = int + (1-est_cdf(lambda_vec[k]))*term1 - (1-est_cdf(lambda_vec[k+1]))*term2
							wfocs_vec[i] = -res
						end
						# lambda FOC
						if i == 1
							term1 = (rho_vec[k] - c)*Lshare(ps1)*M*est_pdf(lambda_vec[k])
							term2a = (ps1 - rho_vec[k] + lambda_vec[k])*Ld_share(ps1)*M*Ld_pstar_d_lambda(rho_vec[k],lambda_vec[k])
							term2b = Lshare(ps1)*M*(Ld_pstar_d_lambda(rho_vec[k],lambda_vec[k]) + 1)
							term2c = (ps2 - rho_vec[k-1] + lambda_vec[k])*Ld_share(ps2)*M*Ld_pstar_d_lambda(rho_vec[k-1],lambda_vec[k])
							term2d = Lshare(ps2)*M*(Ld_pstar_d_lambda(rho_vec[k-1],lambda_vec[k]) + 1)
							term3a = (ps1 - rho_vec[k] + lambda_vec[k])*Lshare(ps1)*M
							term3b = (ps2 - rho_vec[k-1] + lambda_vec[k])*Lshare(ps2)*M
							res = -term1 + (1-est_cdf(lambda_vec[k]))*(term2a + term2b - term2c - term2d) - est_pdf(lambda_vec[k])*(term3a - term3b)
							wfocs_vec[i+N-1] = -res
						else
							term0 = (rho_vec[k-1] - c)*Lshare(ps2)*M
							term1 = (rho_vec[k] - c)*Lshare(ps1)*M
							term2a = (ps1 - rho_vec[k] + lambda_vec[k])*Ld_share(ps1)*M*Ld_pstar_d_lambda(rho_vec[k],lambda_vec[k])
							term2b = Lshare(ps1)*M*(Ld_pstar_d_lambda(rho_vec[k],lambda_vec[k])+1)
							term2c = (ps2 - rho_vec[k-1] + lambda_vec[k])*Ld_share(ps2)*M*Ld_pstar_d_lambda(rho_vec[k-1],lambda_vec[k])
							term2d = Lshare(ps2)*M*(Ld_pstar_d_lambda(rho_vec[k-1],lambda_vec[k]) + 1)
							term3a = (ps1 - rho_vec[k] + lambda_vec[k])*Lshare(ps1)*M
							term3b = (ps2 - rho_vec[k-1] + lambda_vec[k])*Lshare(ps2)*M
							res = est_pdf(lambda_vec[k])*(term0 - term1) + (1-est_cdf(lambda_vec[k]))*(term2a + term2b - term2c - term2d) - est_pdf(lambda_vec[k])*(term3a - term3b)
							wfocs_vec[i+N-1] = -res
						end
						
					end
				end
				function Lwhess!(theta::Vector, whess_mat)
					lambda_vec = [lambda_lb; theta[N:end]; lambda_ub] 
					rho_vec = [rho_0 ; theta[1:N-1]]
					#rho rho part diagonal elements
					for i = 1:N-1
						k = i+1
						f(l) = ((rho_vec[k] - c)*(Ld_share(Lp_star(rho_vec[k],l))*M*Ld2_pstar_d2_rho(rho_vec[k],l) + (Ld_pstar_d_rho(rho_vec[k],l)^2)*Ldd_share(Lp_star(rho_vec[k],l))*M) + 2*Ld_share(Lp_star(rho_vec[k],l))*M*Ld_pstar_d_rho(rho_vec[k],l))*est_pdf(l)
						int = sparse_int(f,lambda_vec[k],lambda_vec[k+1])
						ps1 = Lp_star(rho_vec[k],lambda_vec[k])
						ss1 = (ps1 - rho_vec[k] + lambda_vec[k])*(Ld_share(ps1)*M*Ld2_pstar_d2_rho(rho_vec[k],lambda_vec[k]) + (Ld_pstar_d_rho(rho_vec[k],lambda_vec[k])^2)*Ldd_share(ps1)*M)
						ss2 = Ld_share(ps1)*M*Ld_pstar_d_rho(rho_vec[k],lambda_vec[k])*(Ld_pstar_d_rho(rho_vec[k],lambda_vec[k])-1)
						ss3 = Lshare(ps1)*M*Ld2_pstar_d2_rho(rho_vec[k],lambda_vec[k])
						ss4 = (Ld_pstar_d_rho(rho_vec[k],lambda_vec[k]) - 1)*Ld_share(ps1)*M*Ld_pstar_d_rho(rho_vec[k],lambda_vec[k])
						if i != N-1
							ps2 = Lp_star(rho_vec[k],lambda_vec[k+1])
							ss5 = (ps2 - rho_vec[k] + lambda_vec[k+1])*(Ld_share(ps2)*M*Ld2_pstar_d2_rho(rho_vec[k],lambda_vec[k+1]) + (Ld_pstar_d_rho(rho_vec[k],lambda_vec[k+1])^2)*Ldd_share(ps2)*M)
							ss6 = Ld_share(ps2)*M*Ld_pstar_d_rho(rho_vec[k],lambda_vec[k+1])*(Ld_pstar_d_rho(rho_vec[k],lambda_vec[k+1])-1)
							ss7 = Lshare(ps2)*M*Ld2_pstar_d2_rho(rho_vec[k],lambda_vec[k+1])
							ss8 = (Ld_pstar_d_rho(rho_vec[k],lambda_vec[k+1]) - 1)*Ld_share(ps2)*M*Ld_pstar_d_rho(rho_vec[k],lambda_vec[k+1])
							res = int + (1-est_cdf(lambda_vec[k]))*(ss1 + ss2 + ss3 + ss4) - (1-est_cdf(lambda_vec[k+1]))*(ss5 + ss6 + ss7 + ss8)
							whess_mat[i,i] = -res	
						else
							res = int + (1-est_cdf(lambda_vec[k]))*(ss1 + ss2 + ss3 + ss4)
							whess_mat[i,i] = -res
						end
					end
					# rho rho and lambda lambda off diagonals
					for i = 1:N-1
						for j = 1:N-1
							if j != i
								whess_mat[i,j] = 0.0
								whess_mat[i+N-1,j+N-1] = 0.0
							end
						end
					end
					#lambda lambda part diagonals
					for i = 1:N-1
						k = i+1
						ps1 = Lp_star(rho_vec[k],lambda_vec[k])
						ps2 = Lp_star(rho_vec[k-1],lambda_vec[k])
						ss1 = (rho_vec[k] - c)*Ld_share(ps1)*M*Ld_pstar_d_lambda(rho_vec[k],lambda_vec[k])
						ss2 = (rho_vec[k] - c)*Lshare(ps1)*M
						ss3 = (ps1-rho_vec[k]+lambda_vec[k])*(Ld_share(ps1)*M*Ld2_pstar_d2_lambda(rho_vec[k],lambda_vec[k]) + (Ld_pstar_d_lambda(rho_vec[k],lambda_vec[k])^2)*Ldd_share(ps1)*M)
						ss4 = Ld_share(ps1)*M*Ld_pstar_d_lambda(rho_vec[k],lambda_vec[k])*(Ld_pstar_d_lambda(rho_vec[k],lambda_vec[k])+1)
						ss5 = Lshare(ps1)*M*Ld2_pstar_d2_lambda(rho_vec[k],lambda_vec[k])
						ss6 = (Ld_pstar_d_lambda(rho_vec[k],lambda_vec[k]) + 1)*Ld_share(ps1)*M*Ld_pstar_d_lambda(rho_vec[k],lambda_vec[k])
						ss7 = (ps2-rho_vec[k-1]+lambda_vec[k])*(Ld_share(ps2)*M*Ld2_pstar_d2_lambda(rho_vec[k-1],lambda_vec[k]) + (Ld_pstar_d_lambda(rho_vec[k-1],lambda_vec[k])^2)*Ldd_share(ps2)*M)
						ss8 = Ld_share(ps2)*M*Ld_pstar_d_lambda(rho_vec[k-1],lambda_vec[k])*(Ld_pstar_d_lambda(rho_vec[k-1],lambda_vec[k])+1)
						ss9 = Lshare(ps2)*M*Ld2_pstar_d2_lambda(rho_vec[k-1],lambda_vec[k])
						ss10 = (Ld_pstar_d_lambda(rho_vec[k-1],lambda_vec[k]) + 1)*Ld_share(ps2)*M*Ld_pstar_d_lambda(rho_vec[k-1],lambda_vec[k])
						ss11 = (ps1 - rho_vec[k] + lambda_vec[k])*Ld_share(ps1)*M*Ld_pstar_d_lambda(rho_vec[k],lambda_vec[k])
						ss12 = Lshare(ps1)*M*(Ld_pstar_d_lambda(rho_vec[k],lambda_vec[k]) + 1)
						ss13 = (ps2 - rho_vec[k-1] + lambda_vec[k])*Ld_share(ps2)*M*Ld_pstar_d_lambda(rho_vec[k-1],lambda_vec[k])
						ss14 = Lshare(ps2)*M*(Ld_pstar_d_lambda(rho_vec[k-1],lambda_vec[k]) + 1)
						ss15 = (ps1 - rho_vec[k] + lambda_vec[k])*Lshare(ps1)*M
						ss16 = (ps2 - rho_vec[k-1] + lambda_vec[k])*Lshare(ps2)*M
						ss17 = (rho_vec[k-1] - c)*Ld_share(ps2)*M*Ld_pstar_d_lambda(rho_vec[k-1],lambda_vec[k])
						ss18 = (rho_vec[k-1] - c)*Lshare(ps2)*M

						if i == 1
							res = -(est_pdf(lambda_vec[k])*ss1 + d_est_pdf(lambda_vec[k])*ss2) + (1-est_cdf(lambda_vec[k]))*(ss3 + ss4 + ss5 + ss6 - ss7 - ss8 - ss9 - ss10) - 2*est_pdf(lambda_vec[k])*(ss11 + ss12 - ss13 - ss14) - d_est_pdf(lambda_vec[k])*(ss15 - ss16)
							whess_mat[i+N-1,i+N-1] = -res
						else
							res = est_pdf(lambda_vec[k])*ss17 + d_est_pdf(lambda_vec[k])*ss18 - (est_pdf(lambda_vec[k])*ss1 + d_est_pdf(lambda_vec[k])*ss2) + (1-est_cdf(lambda_vec[k]))*(ss3 + ss4 + ss5 + ss6 - ss7 - ss8 - ss9 - ss10) - 2*est_pdf(lambda_vec[k])*(ss11 + ss12 - ss13 - ss14) - d_est_pdf(lambda_vec[k])*(ss15 - ss16)
							whess_mat[i+N-1,i+N-1] = -res
						end
					end
					# lambda rho part = transposed rho lambda part
					for i = 1:N-1
						for j = 1:N-1
							k = i+1
							if j == i
								ps1 = Lp_star(rho_vec[k],lambda_vec[k])
								ss1 = (rho_vec[k] - c)*Ld_share(ps1)*M*Ld_pstar_d_rho(rho_vec[k],lambda_vec[k])
								ss2 = Lshare(ps1)*M
								ss3 = (ps1 - rho_vec[k] + lambda_vec[k])*(Ld_share(ps1)*M*Ld2_pstar_d_rho_d_lambda(rho_vec[k],lambda_vec[k]) + Ld_pstar_d_lambda(rho_vec[k],lambda_vec[k])*Ldd_share(ps1)*M*Ld_pstar_d_rho(rho_vec[k],lambda_vec[k]))
								ss4 = Ld_share(ps1)*M*Ld_pstar_d_lambda(rho_vec[k],lambda_vec[k])*(Ld_pstar_d_rho(rho_vec[k],lambda_vec[k]) - 1)
								ss5 = Lshare(ps1)*M*Ld2_pstar_d_rho_d_lambda(rho_vec[k],lambda_vec[k])
								ss6 = (Ld_pstar_d_lambda(rho_vec[k],lambda_vec[k]) + 1)*Ld_share(ps1)*M*Ld_pstar_d_rho(rho_vec[k],lambda_vec[k])
								ss7 = (ps1 - rho_vec[k] + lambda_vec[k])*Ld_share(ps1)*M*Ld_pstar_d_rho(rho_vec[k],lambda_vec[k])
								ss8 = Lshare(ps1)*M*(Ld_pstar_d_rho(rho_vec[k],lambda_vec[k]) - 1)
								res = -est_pdf(lambda_vec[k])*(ss1 + ss2) + (1-est_cdf(lambda_vec[k]))*(ss3 + ss4 + ss5 + ss6) - est_pdf(lambda_vec[k])*(ss7 + ss8)
								whess_mat[i+N-1,j] = -res
								whess_mat[j,i+N-1] = -res
							elseif j == i-1
								ps2 = Lp_star(rho_vec[k-1],lambda_vec[k])
								ts1 = (rho_vec[k-1] - c)*Ld_share(ps2)*M*Ld_pstar_d_rho(rho_vec[k-1],lambda_vec[k])
								ts2 = Lshare(ps2)*M
								ts3 = (ps2 - rho_vec[k-1] + lambda_vec[k])*(Ld_share(ps2)*M*Ld2_pstar_d_rho_d_lambda(rho_vec[k-1],lambda_vec[k]) + Ld_pstar_d_lambda(rho_vec[k-1],lambda_vec[k])*Ldd_share(ps2)*M*Ld_pstar_d_rho(rho_vec[k-1],lambda_vec[k]))
								ts4 = Ld_share(ps2)*M*Ld_pstar_d_lambda(rho_vec[k-1],lambda_vec[k])*(Ld_pstar_d_rho(rho_vec[k-1],lambda_vec[k]) - 1)
								ts5 = Lshare(ps2)*M*Ld2_pstar_d_rho_d_lambda(rho_vec[k-1],lambda_vec[k])
								ts6 = (Ld_pstar_d_lambda(rho_vec[k-1],lambda_vec[k]) + 1)*Ld_share(ps2)*M*Ld_pstar_d_rho(rho_vec[k-1],lambda_vec[k])
								ts7 = (ps2 - rho_vec[k-1] + lambda_vec[k])*Ld_share(ps2)*M*Ld_pstar_d_rho(rho_vec[k-1],lambda_vec[k])
								ts8 = Lshare(ps2)*M*(Ld_pstar_d_rho(rho_vec[k-1],lambda_vec[k]) - 1)
								res = est_pdf(lambda_vec[k])*(ts1 + ts2) - (1-est_cdf(lambda_vec[k]))*(ts3 + ts4 + ts5 + ts6) + est_pdf(lambda_vec[k])*(ts7 + ts8)
								whess_mat[i+N-1,j] = -res
								whess_mat[j,i+N-1] = -res
							else
								whess_mat[i+N-1,j] = 0.0
								whess_mat[j,i+N-1] = 0.0
							end
						end
					end
				end
				rho_start = convert(Array{Float64,1},[urho(k) for k = 1:N-1])
				lambda_start = convert(Array{Float64,1},[ulambda(k) for k = 1:N-1])	
				innerx0 = [rho_start;lambda_start]
				#innerx0 = [1.0, 0.75, 0.5, 0.3, 0.6, 0.9] + 0*randn(2*(N-1))
				# checking hessian and gradient
				#=	
				println(innerx0)
				
				gtest1 = ones(2*(N-1))
				gtest2 = ones(2*(N-1))
				htest = ones(2*(N-1),2*(N-1))
				eps = zeros(2*(N-1)) 
				step = 1e-9
				eps[6] = step
				est_grad = (Lw_profit(innerx0+eps) - Lw_profit(innerx0-eps))/(2*step)
				println("Numerical grad: ", est_grad)
				Lwfocs!(innerx0,gtest1)
				println(gtest1)

				Lwfocs!(innerx0+eps,gtest1)
				Lwfocs!(innerx0-eps,gtest2)
				Lwhess!(innerx0,htest)
				println((gtest1 - gtest2)/(2*step))
				println(htest)
				solution = 1
				return solution
				=#	
				solution = optimize(Lw_profit,Lwfocs!,Lwhess!,innerx0,method=NewtonTrustRegion(), show_trace = false, extended_trace = false, iterations = 1500, f_tol = 1e-64, g_tol = 1e-6)
				#println(solution)
				est_rhos = [rho_0 ; Optim.minimizer(solution)[1:N-1]]
				est_lambdas = [lambda_lb ; Optim.minimizer(solution)[N:end] ; lambda_ub]

				
				# Calculating fixed fees
				est_ff = [0.0]
				for i in 1:N-1
					k = i+1
					A = est_ff[k-1] + (Lp_star(est_rhos[k],est_lambdas[k]) - est_rhos[k] + est_lambdas[k])*Lshare(Lp_star(est_rhos[k],est_lambdas[k]))*M - (Lp_star(est_rhos[k-1],est_lambdas[k]) - est_rhos[k-1] + est_lambdas[k])*Lshare(Lp_star(est_rhos[k-1],est_lambdas[k]))*M
					push!(est_ff,A)
				end
				return (est_rhos,est_ff,est_lambdas)
				
			end
			function price_sched_calc(params,N;hot_start=nothing)
				# params are the params of the wholesaler's problem we're trying to
				# estimate. 

				# Function returns the coefficients defining the price schedule as [rho, lambda]
				# where rho is n long and lambda is n-1 long for an n option schedule

				c = params[1] # marginal cost for wholesaler
				#max_mc = params[2] # max MC for retailer. Scales type parameter
				#M = exp(params[5])
				#distribution parameters
				a = exp(params[2]) #
				b = exp(params[3]) # 
				lambda_lb = 0 #-max_mc
				lambda_ub = max_mc
					
				est_cdf(x) = cdf(Beta(a,b),(x-lambda_lb)/(lambda_ub - lambda_lb))
				est_pdf(x) = pdf(Beta(a,b),(x-lambda_lb)/(lambda_ub - lambda_lb))/(lambda_ub - lambda_lb)
				d_est_pdf(x) = ((1/(lambda_ub - lambda_lb))^2)*((a + b - 2)*(x-lambda_lb)/(lambda_ub - lambda_lb) - (a - 1))/(((x-lambda_lb)/(lambda_ub - lambda_lb) -1)*(x-lambda_lb)/(lambda_ub - lambda_lb))*pdf(Beta(a,b),(x-lambda_lb)/(lambda_ub - lambda_lb))
				
				#=
				#= below formulation of th Kumaraswarmy distribution isn't exactly right. Should only be defined on (0,1) and pdf is 0 outside that. Programming that in makes it act funny =#		
				est_cdf(x) = 1.0 - (1.0-(x/max_mc)^a)^b
				est_pdf(x) = ForwardDiff.derivative(est_cdf,x)
				d_est_pdf(x) = ForwardDiff.derivative(est_pdf,x)
				=#


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
				#rho_0 = obs_rhos[1]*2.0
				rho_0 = 30.0
				##### Optimizer Approach
				function w_profit(theta)
					lambda_vec = [lambda_lb; theta[N:end]; lambda_ub] 
					rho_vec = [rho_0 ; theta[1:N-1]]
					profit = 0.0
					for i = 1:N-1
						k = i+1 # dealing with indexing
						f(l) =  ((rho_vec[k] - c)*share(p_star(rho_vec[k],l))*M)*est_pdf(l)
						int = sparse_int(f,lambda_vec[k],lambda_vec[k+1])
						# Pre-calculating some stuff to avoid repeated calls to p_star
						ps1 = p_star(rho_vec[k],lambda_vec[k])
						ps2 = p_star(rho_vec[k-1],lambda_vec[k])
						inc = int + (1-est_cdf(lambda_vec[k]))*((ps1 - rho_vec[k] + lambda_vec[k])*share(ps1)*M - (ps2 - rho_vec[k-1] + lambda_vec[k])*share(ps2)*M)
						profit = profit + inc
					end
					return -profit
				end
				function wfocs!(theta::Vector, wfocs_vec)
					lambda_vec = [lambda_lb; theta[N:end]; lambda_ub] 
					rho_vec = [rho_0 ; theta[1:N-1]] # that inital value is just to force share(p_star(rho_0,lambda_0)) = 0. Ideally should be Inf
					# Calculating FOCs
					for i in 1:N-1 
						k = i+1 # dealing with the increment in indexing and making code look more like notation
						#calculating integral for rho FOC
A
						f(l) =  ((rho_vec[k] - c)*d_share(p_star(rho_vec[k],l))*M*d_pstar_d_rho(rho_vec[k],l) + share(p_star(rho_vec[k],l))*M)*est_pdf(l)
						int = sparse_int(f,lambda_vec[k],lambda_vec[k+1])
						# Pre-calculating some stuff to avoid repeated calls to p_star
						ps1 = p_star(rho_vec[k],lambda_vec[k])
						ps2 = p_star(rho_vec[k-1],lambda_vec[k])
						ps3 = p_star(rho_vec[k],lambda_vec[k+1])
						#rho FOC
						if i == N-1
							term1 = (ps1 - rho_vec[k] + lambda_vec[k])*d_share(ps1)*M*d_pstar_d_rho(rho_vec[k],lambda_vec[k]) + share(ps1)*M*(d_pstar_d_rho(rho_vec[k],lambda_vec[k]) - 1)
							
							res = int + (1-est_cdf(lambda_vec[k]))*term1
							wfocs_vec[i] = -res
						else
							term1 = (ps1 - rho_vec[k] + lambda_vec[k])*d_share(ps1)*M*d_pstar_d_rho(rho_vec[k],lambda_vec[k]) + share(ps1)*M*(d_pstar_d_rho(rho_vec[k],lambda_vec[k]) - 1)
							term2 = (ps3 - rho_vec[k] + lambda_vec[k+1])*d_share(ps3)*M*d_pstar_d_rho(rho_vec[k],lambda_vec[k+1]) + share(ps3)*M*(d_pstar_d_rho(rho_vec[k],lambda_vec[k+1]) - 1)
							
							res = int + (1-est_cdf(lambda_vec[k]))*term1 - (1-est_cdf(lambda_vec[k+1]))*term2
							wfocs_vec[i] = -res
						end
						# lambda FOC
						if i == 1
							term1 = (rho_vec[k] - c)*share(ps1)*M*est_pdf(lambda_vec[k])
							term2a = (ps1 - rho_vec[k] + lambda_vec[k])*d_share(ps1)*M*d_pstar_d_lambda(rho_vec[k],lambda_vec[k])
							term2b = share(ps1)*M*(d_pstar_d_lambda(rho_vec[k],lambda_vec[k]) + 1)
							term2c = (ps2 - rho_vec[k-1] + lambda_vec[k])*d_share(ps2)*M*d_pstar_d_lambda(rho_vec[k-1],lambda_vec[k])
							term2d = share(ps2)*M*(d_pstar_d_lambda(rho_vec[k-1],lambda_vec[k]) + 1)
							term3a = (ps1 - rho_vec[k] + lambda_vec[k])*share(ps1)*M
							term3b = (ps2 - rho_vec[k-1] + lambda_vec[k])*share(ps2)*M
							res = -term1 + (1-est_cdf(lambda_vec[k]))*(term2a + term2b - term2c - term2d) - est_pdf(lambda_vec[k])*(term3a - term3b)
							wfocs_vec[i+N-1] = -res
						else
							term0 = (rho_vec[k-1] - c)*share(ps2)*M
							term1 = (rho_vec[k] - c)*share(ps1)*M
							term2a = (ps1 - rho_vec[k] + lambda_vec[k])*d_share(ps1)*M*d_pstar_d_lambda(rho_vec[k],lambda_vec[k])
							term2b = share(ps1)*M*(d_pstar_d_lambda(rho_vec[k],lambda_vec[k])+1)
							term2c = (ps2 - rho_vec[k-1] + lambda_vec[k])*d_share(ps2)*M*d_pstar_d_lambda(rho_vec[k-1],lambda_vec[k])
							term2d = share(ps2)*M*(d_pstar_d_lambda(rho_vec[k-1],lambda_vec[k]) + 1)
							term3a = (ps1 - rho_vec[k] + lambda_vec[k])*share(ps1)*M
							term3b = (ps2 - rho_vec[k-1] + lambda_vec[k])*share(ps2)*M
							res = est_pdf(lambda_vec[k])*(term0 - term1) + (1-est_cdf(lambda_vec[k]))*(term2a + term2b - term2c - term2d) - est_pdf(lambda_vec[k])*(term3a - term3b)
							wfocs_vec[i+N-1] = -res
						end
						
					end
				end
				function whess!(theta::Vector, whess_mat)
					lambda_vec = [lambda_lb; theta[N:end]; lambda_ub] 
					rho_vec = [rho_0 ; theta[1:N-1]]
					#rho rho part diagonal elements
					for i = 1:N-1
						k = i+1
						f(l) = ((rho_vec[k] - c)*(d_share(p_star(rho_vec[k],l))*M*d2_pstar_d2_rho(rho_vec[k],l) + (d_pstar_d_rho(rho_vec[k],l)^2)*dd_share(p_star(rho_vec[k],l))*M) + 2*d_share(p_star(rho_vec[k],l))*M*d_pstar_d_rho(rho_vec[k],l))*est_pdf(l)
						int = sparse_int(f,lambda_vec[k],lambda_vec[k+1])
						ps1 = p_star(rho_vec[k],lambda_vec[k])
						ss1 = (ps1 - rho_vec[k] + lambda_vec[k])*(d_share(ps1)*M*d2_pstar_d2_rho(rho_vec[k],lambda_vec[k]) + (d_pstar_d_rho(rho_vec[k],lambda_vec[k])^2)*dd_share(ps1)*M)
						ss2 = d_share(ps1)*M*d_pstar_d_rho(rho_vec[k],lambda_vec[k])*(d_pstar_d_rho(rho_vec[k],lambda_vec[k])-1)
						ss3 = share(ps1)*M*d2_pstar_d2_rho(rho_vec[k],lambda_vec[k])
						ss4 = (d_pstar_d_rho(rho_vec[k],lambda_vec[k]) - 1)*d_share(ps1)*M*d_pstar_d_rho(rho_vec[k],lambda_vec[k])
						if i != N-1
							ps2 = p_star(rho_vec[k],lambda_vec[k+1])
							ss5 = (ps2 - rho_vec[k] + lambda_vec[k+1])*(d_share(ps2)*M*d2_pstar_d2_rho(rho_vec[k],lambda_vec[k+1]) + (d_pstar_d_rho(rho_vec[k],lambda_vec[k+1])^2)*dd_share(ps2)*M)
							ss6 = d_share(ps2)*M*d_pstar_d_rho(rho_vec[k],lambda_vec[k+1])*(d_pstar_d_rho(rho_vec[k],lambda_vec[k+1])-1)
							ss7 = share(ps2)*M*d2_pstar_d2_rho(rho_vec[k],lambda_vec[k+1])
							ss8 = (d_pstar_d_rho(rho_vec[k],lambda_vec[k+1]) - 1)*d_share(ps2)*M*d_pstar_d_rho(rho_vec[k],lambda_vec[k+1])
							res = int + (1-est_cdf(lambda_vec[k]))*(ss1 + ss2 + ss3 + ss4) - (1-est_cdf(lambda_vec[k+1]))*(ss5 + ss6 + ss7 + ss8)
							whess_mat[i,i] = -res	
						else
							res = int + (1-est_cdf(lambda_vec[k]))*(ss1 + ss2 + ss3 + ss4)
							whess_mat[i,i] = -res
						end
					end
					# rho rho and lambda lambda off diagonals
					for i = 1:N-1
						for j = 1:N-1
							if j != i
								whess_mat[i,j] = 0.0
								whess_mat[i+N-1,j+N-1] = 0.0
							end
						end
					end
					#lambda lambda part diagonals
					for i = 1:N-1
						k = i+1
						ps1 = p_star(rho_vec[k],lambda_vec[k])
						ps2 = p_star(rho_vec[k-1],lambda_vec[k])
						ss1 = (rho_vec[k] - c)*d_share(ps1)*M*d_pstar_d_lambda(rho_vec[k],lambda_vec[k])
						ss2 = (rho_vec[k] - c)*share(ps1)*M
						ss3 = (ps1-rho_vec[k]+lambda_vec[k])*(d_share(ps1)*M*d2_pstar_d2_lambda(rho_vec[k],lambda_vec[k]) + (d_pstar_d_lambda(rho_vec[k],lambda_vec[k])^2)*dd_share(ps1)*M)
						ss4 = d_share(ps1)*M*d_pstar_d_lambda(rho_vec[k],lambda_vec[k])*(d_pstar_d_lambda(rho_vec[k],lambda_vec[k])+1)
						ss5 = share(ps1)*M*d2_pstar_d2_lambda(rho_vec[k],lambda_vec[k])
						ss6 = (d_pstar_d_lambda(rho_vec[k],lambda_vec[k]) + 1)*d_share(ps1)*M*d_pstar_d_lambda(rho_vec[k],lambda_vec[k])
						ss7 = (ps2-rho_vec[k-1]+lambda_vec[k])*(d_share(ps2)*M*d2_pstar_d2_lambda(rho_vec[k-1],lambda_vec[k]) + (d_pstar_d_lambda(rho_vec[k-1],lambda_vec[k])^2)*dd_share(ps2)*M)
						ss8 = d_share(ps2)*M*d_pstar_d_lambda(rho_vec[k-1],lambda_vec[k])*(d_pstar_d_lambda(rho_vec[k-1],lambda_vec[k])+1)
						ss9 = share(ps2)*M*d2_pstar_d2_lambda(rho_vec[k-1],lambda_vec[k])
						ss10 = (d_pstar_d_lambda(rho_vec[k-1],lambda_vec[k]) + 1)*d_share(ps2)*M*d_pstar_d_lambda(rho_vec[k-1],lambda_vec[k])
						ss11 = (ps1 - rho_vec[k] + lambda_vec[k])*d_share(ps1)*M*d_pstar_d_lambda(rho_vec[k],lambda_vec[k])
						ss12 = share(ps1)*M*(d_pstar_d_lambda(rho_vec[k],lambda_vec[k]) + 1)
						ss13 = (ps2 - rho_vec[k-1] + lambda_vec[k])*d_share(ps2)*M*d_pstar_d_lambda(rho_vec[k-1],lambda_vec[k])
						ss14 = share(ps2)*M*(d_pstar_d_lambda(rho_vec[k-1],lambda_vec[k]) + 1)
						ss15 = (ps1 - rho_vec[k] + lambda_vec[k])*share(ps1)*M
						ss16 = (ps2 - rho_vec[k-1] + lambda_vec[k])*share(ps2)*M
						ss17 = (rho_vec[k-1] - c)*d_share(ps2)*M*d_pstar_d_lambda(rho_vec[k-1],lambda_vec[k])
						ss18 = (rho_vec[k-1] - c)*share(ps2)*M

						if i == 1
							res = -(est_pdf(lambda_vec[k])*ss1 + d_est_pdf(lambda_vec[k])*ss2) + (1-est_cdf(lambda_vec[k]))*(ss3 + ss4 + ss5 + ss6 - ss7 - ss8 - ss9 - ss10) - 2*est_pdf(lambda_vec[k])*(ss11 + ss12 - ss13 - ss14) - d_est_pdf(lambda_vec[k])*(ss15 - ss16)
							whess_mat[i+N-1,i+N-1] = -res
						else
							res = est_pdf(lambda_vec[k])*ss17 + d_est_pdf(lambda_vec[k])*ss18 - (est_pdf(lambda_vec[k])*ss1 + d_est_pdf(lambda_vec[k])*ss2) + (1-est_cdf(lambda_vec[k]))*(ss3 + ss4 + ss5 + ss6 - ss7 - ss8 - ss9 - ss10) - 2*est_pdf(lambda_vec[k])*(ss11 + ss12 - ss13 - ss14) - d_est_pdf(lambda_vec[k])*(ss15 - ss16)
							whess_mat[i+N-1,i+N-1] = -res
						end
					end
					# lambda rho part = transposed rho lambda part
					for i = 1:N-1
						for j = 1:N-1
							k = i+1
							if j == i
								ps1 = p_star(rho_vec[k],lambda_vec[k])
								ss1 = (rho_vec[k] - c)*d_share(ps1)*M*d_pstar_d_rho(rho_vec[k],lambda_vec[k])
								ss2 = share(ps1)*M
								ss3 = (ps1 - rho_vec[k] + lambda_vec[k])*(d_share(ps1)*M*d2_pstar_d_rho_d_lambda(rho_vec[k],lambda_vec[k]) + d_pstar_d_lambda(rho_vec[k],lambda_vec[k])*dd_share(ps1)*M*d_pstar_d_rho(rho_vec[k],lambda_vec[k]))
								ss4 = d_share(ps1)*M*d_pstar_d_lambda(rho_vec[k],lambda_vec[k])*(d_pstar_d_rho(rho_vec[k],lambda_vec[k]) - 1)
								ss5 = share(ps1)*M*d2_pstar_d_rho_d_lambda(rho_vec[k],lambda_vec[k])
								ss6 = (d_pstar_d_lambda(rho_vec[k],lambda_vec[k]) + 1)*d_share(ps1)*M*d_pstar_d_rho(rho_vec[k],lambda_vec[k])
								ss7 = (ps1 - rho_vec[k] + lambda_vec[k])*d_share(ps1)*M*d_pstar_d_rho(rho_vec[k],lambda_vec[k])
								ss8 = share(ps1)*M*(d_pstar_d_rho(rho_vec[k],lambda_vec[k]) - 1)
								res = -est_pdf(lambda_vec[k])*(ss1 + ss2) + (1-est_cdf(lambda_vec[k]))*(ss3 + ss4 + ss5 + ss6) - est_pdf(lambda_vec[k])*(ss7 + ss8)
								whess_mat[i+N-1,j] = -res
								whess_mat[j,i+N-1] = -res
							elseif j == i-1
								ps2 = p_star(rho_vec[k-1],lambda_vec[k])
								ts1 = (rho_vec[k-1] - c)*d_share(ps2)*M*d_pstar_d_rho(rho_vec[k-1],lambda_vec[k])
								ts2 = share(ps2)*M
								ts3 = (ps2 - rho_vec[k-1] + lambda_vec[k])*(d_share(ps2)*M*d2_pstar_d_rho_d_lambda(rho_vec[k-1],lambda_vec[k]) + d_pstar_d_lambda(rho_vec[k-1],lambda_vec[k])*dd_share(ps2)*M*d_pstar_d_rho(rho_vec[k-1],lambda_vec[k]))
								ts4 = d_share(ps2)*M*d_pstar_d_lambda(rho_vec[k-1],lambda_vec[k])*(d_pstar_d_rho(rho_vec[k-1],lambda_vec[k]) - 1)
								ts5 = share(ps2)*M*d2_pstar_d_rho_d_lambda(rho_vec[k-1],lambda_vec[k])
								ts6 = (d_pstar_d_lambda(rho_vec[k-1],lambda_vec[k]) + 1)*d_share(ps2)*M*d_pstar_d_rho(rho_vec[k-1],lambda_vec[k])
								ts7 = (ps2 - rho_vec[k-1] + lambda_vec[k])*d_share(ps2)*M*d_pstar_d_rho(rho_vec[k-1],lambda_vec[k])
								ts8 = share(ps2)*M*(d_pstar_d_rho(rho_vec[k-1],lambda_vec[k]) - 1)
								res = est_pdf(lambda_vec[k])*(ts1 + ts2) - (1-est_cdf(lambda_vec[k]))*(ts3 + ts4 + ts5 + ts6) + est_pdf(lambda_vec[k])*(ts7 + ts8)
								whess_mat[i+N-1,j] = -res
								whess_mat[j,i+N-1] = -res
							else
								whess_mat[i+N-1,j] = 0.0
								whess_mat[j,i+N-1] = 0.0
							end
						end
					end
				end
				if hot_start == nothing
					rho_start = [max_mc/i for i = 1:N-1]
					lambda_start = [(lambda_ub -lambda_lb)/(N)*i for i = 1:N-1] 	
					innerx0 = [rho_start;lambda_start]
				else
					innerx0 = hot_start
				end
				# checking hessian and gradient
				#=	
				println(x0)
				gtest1 = ones(2*(N-1))
				gtest2 = ones(2*(N-1))
				htest = ones(2*(N-1),2*(N-1))
				eps = zeros(2*(N-1)) 
				step = 1e-9
				eps[8] = step
				est_grad = (w_profit(innerx0+eps) - w_profit(innerx0-eps))/(2*step)
				println("Numerical grad: ", est_grad)
				wfocs!(innerx0,gtest1)
				println(gtest1)

				wfocs!(innerx0+eps,gtest1)
				wfocs!(innerx0-eps,gtest2)
				whess!(innerx0,htest)
				println((gtest1 - gtest2)/(2*step))
				println(htest)
				solution = 1
				=#
				solution = optimize(w_profit,wfocs!,whess!,innerx0,method=NewtonTrustRegion(), show_trace = false, extended_trace = false, iterations = 500, f_tol = 1e-32, g_tol = 1e-6)
				est_rhos = [rho_0 ; Optim.minimizer(solution)[1:N-1]]
				est_lambdas = [lambda_lb ; Optim.minimizer(solution)[N:end] ; lambda_ub]
				#println(solution)
				
				# Calculating fixed fees
				est_ff = [0.0]
				for i in 1:N-1
					k = i+1
					A = est_ff[k-1] + (p_star(est_rhos[k],est_lambdas[k]) - est_rhos[k] + est_lambdas[k])*share(p_star(est_rhos[k],est_lambdas[k]))*M - (p_star(est_rhos[k-1],est_lambdas[k]) - est_rhos[k-1] + est_lambdas[k])*share(p_star(est_rhos[k-1],est_lambdas[k]))*M
					push!(est_ff,A)
				end
				return (est_rhos,est_ff,est_lambdas)
			end
			
			function obj_func(omega::Vector, N::Int, W::Matrix)
				tic()
				hsrho,hsff,hslambda = Lprice_sched_calc(omega,N)
				hs = [hsrho[2:end];hslambda[2:end-1]]
				rho_hat,ff_hat,lambda_hat = price_sched_calc(omega,N; hot_start = hs)
				vec = [(rho_hat[2:end] - obs_rhos) ; (ff_hat[3:end] - obs_ff[2:end])]'
				res = vec*W*vec'
				toc()
				return res[1]
			end
			function Lobj_func(omega::Vector, N::Int, W::Matrix)
				rho_hat,ff_hat,lambda_hat = Lprice_sched_calc(omega,N)
				vec = [(rho_hat[2:end] - obs_rhos) ; (ff_hat[3:end] - obs_ff[2:end])]'
				res = vec*W*vec'
				return res[1]
			end

			function uobj_func(omega::Vector, N::Int, W::Matrix)
				# Omega should just be 2 long
				hsrho,hsff,hslambda = Lprice_sched_calc([omega;log(1.0);log(1.0)],N)
				hs = [hsrho[2:end],hslambda[2:end-1]]
				rho_hat,ff_hat,lambda_hat = price_sched_calc([omega;log(1.0);log(1.0)],N)
				vec = [(rho_hat[2:end] - obs_rhos) ; (ff_hat[3:end] - obs_ff[2:end])]'
				res = vec*W*vec'
				return res[1]
			end
			function uLobj_func(omega::Vector, N::Int, W::Matrix)
				# Omega should just be 2 long
				rho_hat,ff_hat,lambda_hat = Lprice_sched_calc([omega;log(1.0);log(1.0)],N)
				vec = [(rho_hat[2:end] - obs_rhos) ; (ff_hat[3:end] - obs_ff[2:end])]'
				res = vec*W*vec'
				return res[1]
			end
			function uLobj_foc!(omega::Vector,obj_foc_mat,N::Integer,W::Matrix)
				#column 1 is c, columns 2 is lambda ub
				c = omega[1]
				lambda_ub = omega[2]
				temp_mat = zeros(2*N-3,2)
				# A part for c
				k = 1
				A1 = -M*((LA + LB*(-LA/LB + 2*(c + LA/LB - lambda_ub)/(3-4*N)*(1-2*k)))*(1-2*k)/(3-4*N) - (LA + LB*(-LA/LB + 2*(c + LA/LB - lambda_ub)/(3-4*N)*(2-2*k)))*(2-2*k)/(3-4*N))
				k = 2
				temp_mat[k+N-2,1] = -M*((LA + LB*(-LA/LB + 2*(c + LA/LB - lambda_ub)/(3-4*N)*(1-2*k)))*(1-2*k)/(3-4*N) - (LA + LB*(-LA/LB + 2*(c + LA/LB - lambda_ub)/(3-4*N)*(2-2*k)))*(2-2*k)/(3-4*N)) + A1
				for k = 3:N-1
					temp_mat[k+N-2,1] = -M*((LA + LB*(-LA/LB + 2*(c + LA/LB - lambda_ub)/(3-4*N)*(1-2*k)))*(1-2*k)/(3-4*N) - (LA + LB*(-LA/LB + 2*(c + LA/LB - lambda_ub)/(3-4*N)*(2-2*k)))*(2-2*k)/(3-4*N)) + temp_mat[k+N-3,1]
				end
				# A part for lambda_ub
				k = 1
				A1 = M*((LA + LB*(-LA/LB + 2*(c + LA/LB - lambda_ub)/(3-4*N)*(1-2*k)))*(1-2*k)/(3-4*N) - (LA + LB*(-LA/LB + 2*(c + LA/LB - lambda_ub)/(3-4*N)*(2-2*k)))*(2-2*k)/(3-4*N))
				k = 2
				temp_mat[k+N-2,2] = M*((LA + LB*(-LA/LB + 2*(c + LA/LB - lambda_ub)/(3-4*N)*(1-2*k)))*(1-2*k)/(3-4*N) - (LA + LB*(-LA/LB + 2*(c + LA/LB - lambda_ub)/(3-4*N)*(2-2*k)))*(2-2*k)/(3-4*N)) + A1
				for k = 3:N-1
					temp_mat[k+N-2,2] = M*((LA + LB*(-LA/LB + 2*(c + LA/LB - lambda_ub)/(3-4*N)*(1-2*k)))*(1-2*k)/(3-4*N) - (LA + LB*(-LA/LB + 2*(c + LA/LB - lambda_ub)/(3-4*N)*(2-2*k)))*(2-2*k)/(3-4*N)) + temp_mat[k+N-3,2]
				end
				#rho part for c
				for k = 1:N-1
					temp_mat[k,1] = 2*(1-N-k)/(3-4*N)
				end
				#rho part for lambda_ub
				for k = 1:N-1
					temp_mat[k,2] = 1 - 2*(1-N-k)/(3-4*N)
				end
				#= above calculations produce G(omega) but need:
					2G(omega)'Wg'
				so we need to calculate the outcomes of relevance again=#
				foc_ff_hat = zeros(N-1+1)
				for k = 1:N-1
					foc_ff_hat[k+1] = -M/(4*LB)*((LA + LB*(-LA/LB + 2*(c + LA/LB - lambda_ub)/(3-4*N)*(1-2*k)))^2 - (LA + LB*(-LA/LB + 2*(c + LA/LB - lambda_ub)/(3-4*N)*(2-2*k)))^2) + foc_ff_hat[k+1-1]
				end
				foc_rho_hat = zeros(N-1+1)
				for k = 0:N-1
					foc_rho_hat[k+1] = -LA/LB + lambda_ub + 2*(c + LA/LB - lambda_ub)/(3-4*N)*(1 - N - k)
				end
				vec = [(foc_rho_hat[2:end] - obs_rhos) ; (foc_ff_hat[3:end] - obs_ff[2:end])]'
				res = (2.0*temp_mat'*W*vec')
				for i in 1:size(res)[1] # rows
					for j in 1:size(res)[2] # column
						obj_foc_mat[i,j] = res[i,j]
					end
				end
				
			end

			N = length(obs_rhos)+1
			#W = eye(2*(N-1)- 1)
			
			# testing recovery of params with fake data
			#=
			println("Testing recovery of parameters with 'fake' data")
			x0 = [3.0; log(2.0); log(2.0)]
			hsrho,hsff,hslambda = Lprice_sched_calc(x0,N)
			hs = [hsrho[2:end],hslambda[2:end-1]]
			nlrho,nlff,nllamb = price_sched_calc(x0,N, hot_start = hs)
			obs_rhos = nlrho[2:end]
			obs_ff = [0.0;nlff[3:end]]
			=#
			#ux0 = [3.0,15.0]
			W = Diagonal([1./(obs_rhos.^2) ; 1./(obs_ff[2:end].^2)])*eye(2*N-3)
			# checking Objective func gradient
			#eps = zeros(2)
			#eps[1] = 1e-9
			#println((uLobj_func(ux0+eps,N,W) - uLobj_func(ux0-eps,N,W))/(2*1e-9))
			#jtest = zeros(2,1)
			#uLobj_foc!(ux0,jtest,N,W)	
			#println(jtest)
			
			# testing hot start
			#x0 = [0.0; 2.0; log(1.0); log(1.0)]
			#hsrho,hsff,hslambda = Lprice_sched_calc(x0,N)
			#hs = [hsrho[2:end],hslambda[2:end-1]]
			#println(Lprice_sched_calc(x0,5))
			#println(price_sched_calc(x0,4))
			#println(price_sched_calc(x0,N; hot_start = hs))
			
			# Optimizing uniform linear model	
			#uLg(x) = uLobj_func(x,N,W)
			#focs!(x,vec) = uLobj_foc!(x,vec,N,W)
			#optim_res = optimize(uLg,focs!,ux0,BFGS(),OptimizationOptions(show_every = false, extended_trace = false, iterations = 1500, g_tol = 1e-6))
			#println(optim_res)
			#min_X = Optim.minimizer(optim_res)
			
			outerg(x) = obj_func(x,N,W)
				
			# grid search
			#grid_points = [[i,j,0.0,0.0] for i = 0.0:10.0, j = 2.0:2.0:20.0]
			grid_points = [[i,k,l] for i = linspace(0,40,20),  k = linspace(0,2,3), l=linspace(0,2,3)]
			grid_evals = map(outerg,grid_points)
			min_ind = indmin(grid_evals)
			x0 = grid_points[min_ind]
			println("Best Starting Guess from grid: ",x0)
			
			#=	
			for i = 0.0:10.0
				for j = 2.0:2.0:20.0
					x0 = [i;j;0.0;0.0]
					println(x0)
					println(outerg(x0))
				end
			end
			=#
			outerg(x) = obj_func(x,N,W)
			#x0 = [2.0; 11.0; log(1.0); log(1.0)]
			optim_res = optimize(outerg,x0,NelderMead(),OptimizationOptions(show_every = false, extended_trace = false, iterations = 1500, g_tol = 1e-6))
			println(optim_res)
			min_X = Optim.minimizer(optim_res)
			hsrho,hsff,hslambda = Lprice_sched_calc(min_X,N)
			hs = [hsrho[2:end];hslambda[2:end-1]]
			fit_ps = price_sched_calc(min_X,N, hot_start = hs)

			outtuple = (product,market,min_X[1], max_mc, min_X[2], min_X[3], fit_ps)
			write(csvfile,join(outtuple,"|"),"\n")
			flush(csvfile) # ensures write is written now, so if loop fails, still get some

		else
			println("Product has no matching price data.")
		end
	end
end

close(csvfile)
