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
	 weights = [.0001816, .00063259999999999998, .0012895000000000001, .0021088000000000001, .0030577999999999998, .0041114999999999997, .0052490999999999996, .0064519, .0077034, .0089893000000000004, .0102971, .0116157, .0129348, .0142449, .0155368, .016801900000000002, .018032200000000002, .019219900000000002, .020357799999999999, .021439, .022457299999999999, .023406799999999998, .0242822, .0250786, .025791600000000001, .0264175, .0269527, .027394600000000002, .0277407, .027989199999999999, .028138799999999999, .0281888, .028138799999999999, .027989199999999999, .0277407, .027394600000000002, .0269527, .0264175, .025791600000000001, .0250786, .0242822, .023406799999999998, .022457299999999999, .021439, .020357799999999999, .019219900000000002, .018032200000000002, .016801900000000002, .0155368, .0142449, .0129348, .0116157, .0102971, .0089893000000000004, .0077034, .0064519, .0052490999999999996, .0041114999999999997, .0030577999999999998, .0021088000000000001, .0012895000000000001, .00063259999999999998, .0001816]

	nodes = [.00006349999999999412, .00045090000000003183, .0013969000000000342, .0030839999999999756, .0056576000000000404, .009234399999999976, .013908600000000049, .01975439999999995, .02682859999999998, .035172599999999998, .044814400000000032, .055770399999999998, .068046000000000051, .08163699999999996, .096529700000000052, .11270170000000002, .13012199999999996, .14875190000000005, .16854519999999995, .18944850000000002, .21140210000000004, .23434010000000005, .25819099999999995, .28287810000000002, .30832029999999999, .33443230000000002, .36112509999999998, .3883067, .41588230000000004, .44375549999999997, .47182789999999997, .5, .52817210000000003, .55624450000000003, .58411769999999996, .6116933, .63887490000000002, .66556769999999998, .69167970000000001, .71712189999999998, .74180900000000005, .76565989999999995, .78859789999999996, .81055149999999998, .83145480000000005, .85124809999999995, .86987800000000004, .88729829999999998, .90347029999999995, .91836300000000004, .93195399999999995, .9442296, .95518559999999997, .9648274, .97317140000000002, .98024560000000005, .98609139999999995, .99076560000000002, .99434239999999996, .99691600000000002, .99860309999999997, .99954909999999997, .99993650000000001]

	f_evals = [f((b-a)*u + a) for u in nodes]
	return dot(f_evals, weights)*(b-a)
end

# defining limits of type distribution. Should be consistent with whatever PDF is used
lambda_lb = 0
lambda_ub = 1



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
			#M = df[prod_bool,:M][1]
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
				M = exp(params[5])
				est_pdf(x) = pdf(Beta(a,b),x/max_mc)

				#Defining p-star function. Requires solving NL equation
			#=	function p_star(rho::Number,l::Number)
					function g!(p,gvec)
						gvec[1] =  (p - rho - l)*d_share(p) + share(p)
					end
					
					res = nlsolve(g!,[rho+l]) 
					return res.zero[1]
				end


				#=Interpolationg p_star at a bunch of points to help speed up calculation =#
				p_star_grid = [p_star(i,j) for i = 1.0:100.0, j = 1.0:100.0]
				p_star_grid = convert(Array{Float64,2},p_star_grid)
				p_star_itp = interpolate(p_star_grid, BSpline(Quadratic(Reflect())), OnGrid())
				
				# Derivative of the p_star function. Need to use central difference approx
				function d_pstar_d_rho(rho,lambda) # note that this is the same as d_pstar_d_lambda
					res = d_share(p_star_itp[rho,lambda]) / (dd_share(p_star_itp[rho,lambda])*(p_star_itp[rho,lambda] - rho - lambda) + 2*d_share(p_star_itp[rho,lambda]))
					return res[1]
				end
				function d_pstar_d_lambda(rho,lambda) # see above
					return d_pstar_d_rho(rho,lambda)
				end
				function d2_pstar_d2_rho(rho,lambda)
					num1 = (dd_share(p_star_itp[rho,lambda])*(p_star_itp[rho,lambda] - rho - lambda) + 2*d_share(p_star_itp[rho,lambda]))
					num2 = d_share(p_star_itp[rho,lambda])*(dd_share(p_star_itp[rho,lambda])*(d_pstar_d_rho(rho,lambda) - 1) + ((p_star_itp[rho,lambda] - rho - lambda)*ddd_share(p_star_itp[rho,lambda]) + 2*dd_share(p_star_itp[rho,lambda]))*d_pstar_d_rho(rho,lambda))
					den = dd_share(p_star_itp[rho,lambda])*(p_star_itp[rho,lambda] - rho - lambda) + 2*d_share(p_star_itp[rho,lambda])
					res = (num1 - num2)/(den^2)
					return res
				end
				=#	
				
				##### CODE THAT USES NLSOLVE #####

				# Defining Wholesaler FOCs which can then be solved. Following the syntax to use
				# NLsovle package.
				# NOTE: Can refomulate as constrained minimization if this doesn't work

				function wfocs!(theta::Vector, wfocs_vec)
					nfocs = length(theta) # number of FOCs (last param is wholesaler cost)
					n = round(Int,(nfocs + 1)/2) # figuring out how many parts the price schedule has
					#println("Solving for $n part price schedule")
									
					lambda_vec = [lambda_lb ; theta[n+1:end] ; lambda_ub]
					rho_vec = theta[1:n]
					for i in 1:length(rho_vec)
						f(l) =  ((rho_vec[i] - c)*d_share(p_star(rho_vec[i],l))*d_pstar_d_rho(rho_vec[i],l) + share(p_star(rho_vec[i],l)))*est_pdf(l)
						#wfocs_vec[i] = quadgk(f,lambda_vec[i], lambda_vec[i+1]; abstol= 1e-16)[1]
						wfocs_vec[i] = sparse_int(f,lambda_vec[i], lambda_vec[i+1])
					end
					for i in 1:(length(lambda_vec)-2) # indexing is inclusive so x:x = x for any x
						u1 = p_star(rho_vec[i],lambda_vec[i+1])
						u2 = p_star(rho_vec[i+1],lambda_vec[i+1])
						wfocs_vec[i + length(rho_vec)] = share(u1)*(u1 - lambda_vec[i+1] - c) - share(u2)*(u2 - rho_vec[i+1] - lambda_vec[i+1] + rho_vec[i] - c)
					end
					print(".") # print . for each eval of wfocs
				end
				function wfocs_jac!(theta::Vector, wfocs_jac_vec)
					nfocs = length(theta)
					n = round(Int,(nfocs+1)/2)
					lambda_vec = [lambda_lb ; theta[n+1:end] ; lambda_ub]
					rho_vec = theta[1:n]
					# rho-rho part of Jacobian
					for k in 1:length(rho_vec)
						for j in 1:length(rho_vec)
							if j==k
								f(l) = ((rho_vec[k] - c)*(d_share(p_star(rho_vec[k],l))*d2_pstar_d2_rho(rho_vec[k],l) + (d_pstar_d_rho(rho_vec[k], l)^2)*dd_share(p_star(rho_vec[k],l))) + d_share(p_star(rho_vec[k],l))*d_pstar_d_rho(rho_vec[k],l)*2)*est_pdf(l)
								wfocs_jac_vec[k,j] = (sparse_int(f,lambda_vec[k],lambda_vec[k+1]))
							else
								wfocs_jac_vec[k,j] = 0
							end
						end
					end
					#rho-lambda part of Jacobian
					for k in 1:length(rho_vec)
						for j in 2:(length(lambda_vec)-1) # different range to line up j,k correctly
							if j == k
								u = p_star(rho_vec[k],lambda_vec[j])
								wfocs_jac_vec[k,j+(length(rho_vec)-1)] = (-((rho_vec[k] - c)*d_share(u)*d_pstar_d_lambda(rho_vec[k],lambda_vec[k]) + share(u))*est_pdf(lambda_vec[j]))
							elseif j == k+1
								u = p_star(rho_vec[k],lambda_vec[j])
								wfocs_jac_vec[k,j+(length(rho_vec)-1)] = (((rho_vec[k] - c)*d_share(u)*d_pstar_d_lambda(rho_vec[k],lambda_vec[k]) + share(u))*est_pdf(lambda_vec[j]))
							else
								wfocs_jac_vec[k,j+(length(rho_vec)-1)] = 0
							end
						end
					end
					#lambda-rho part of Jacobian
					for k in 2:(length(lambda_vec)-1)
						for j in 1:length(rho_vec)
							if j == k-1
								u = p_star(rho_vec[j],lambda_vec[k])
								wfocs_jac_vec[k+(length(rho_vec)-1),j]= (share(u)*d_pstar_d_rho(rho_vec[j],lambda_vec[k]) + (u - lambda_vec[k] - c)*d_share(u)*d_pstar_d_rho(rho_vec[j],lambda_vec[k]) - share(p_star(rho_vec[j+1],lambda_vec[k])))
							elseif j == k
								u = p_star(rho_vec[j],lambda_vec[k])
								wfocs_jac_vec[k+(length(rho_vec)-1),j] = -(share(u)*(d_pstar_d_rho(rho_vec[j],lambda_vec[k]) - 1.0) + (u - rho_vec[j] - lambda_vec[k] + rho_vec[j-1] - c)*d_share(u)*d_pstar_d_rho(rho_vec[j],lambda_vec[k]))
							else
								wfocs_jac_vec[k+(length(rho_vec)-1),j] = 0
							end
						end
					end
					#lambda-lambda part of Jacobian
					for k in 2:(length(lambda_vec)-1)
						for j in 2:(length(lambda_vec)-1)
							if j == k
								u1 = p_star(rho_vec[k-1],lambda_vec[k])
								u2 = p_star(rho_vec[k],lambda_vec[k])
								wfocs_jac_vec[k+(length(rho_vec)-1),j+(length(rho_vec)-1)] = (share(u1)*(d_pstar_d_lambda(rho_vec[k-1],lambda_vec[k]) - 1) + (u1 - lambda_vec[k] - c)*d_share(u1)*d_pstar_d_lambda(rho_vec[k-1],lambda_vec[k]) - (share(u2)*(d_pstar_d_lambda(rho_vec[k],lambda_vec[k]) - 1) + (u2 - rho_vec[k] - lambda_vec[k] + rho_vec[k-1] - c)*d_share(u2)*d_pstar_d_lambda(rho_vec[k],lambda_vec[k])))
							else
								wfocs_jac_vec[k+(length(rho_vec)-1),j+(length(rho_vec)-1)] = 0
							end
						end
					end 
				end 
				srand(1987) #reseeding to x0 is same every time
				x0 = ones(2*N-1) + randn(2*N-1)
				#solution = nlsolve(wfocs!,x0,method=:trust_region, show_trace = true, ftol = 1e-8, extended_trace = true, iterations = 1000)
				solution = nlsolve(wfocs!,wfocs_jac!,x0,method=:trust_region, show_trace = false, ftol = 1e-8, extended_trace = false, iterations = 1000)
				est_rhos = solution.zero[1:N] # need to return this. These are the prices for the price schedule
				est_lambdas = [lambda_lb; solution.zero[N+1:end]; lambda_ub] # remaining parameters are lambda cutoffs
				
				# Calculating fixed fees
				est_ff = [0.0]
				for k in 2:N
					A = (p_star(est_rhos[k],est_lambdas[k]) - est_rhos[k] - est_lambdas[k])*share(p_star(est_rhos[k], est_lambdas[k]))*M - (p_star(est_rhos[k-1], est_lambdas[k]) - est_rhos[k-1] - est_lambdas[k])*share(p_star(est_rhos[k-1], est_lambdas[k]))*M + est_ff[k-1]
					push!(est_ff,A)
				end
				return (est_rhos,est_ff,est_lambdas)
			end
			
			function obj_func(omega::Vector, N::Int, W::Matrix)
				tic()
				rho_hat,ff_hat,lambda_hat = price_sched_calc(omega,N)
				vec = [(rho_hat - obs_rhos) ; (ff_hat - obs_ff)]'
				res = vec*W*vec'
				println(res)
				toc()
				return res[1]
			end
		
			N = 4
			W = eye(2*N)
			x0 = [17.0, 13.0, 0.0, 0.0, 6.0]
			g(x) = obj_func(x,N,W)
			optim_res = optimize(g,x0,SimulatedAnnealing(),OptimizationOptions(show_every = true, iterations = 1000))
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
