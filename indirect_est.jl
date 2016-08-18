using DataArrays, DataFrames, ForwardDiff, NLsolve, Roots, Interpolations, JuMP, NLopt

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
burr_cdf1(x) = burr_cdf(x,.1)

burr_pdf(x) = ForwardDiff.derivative(burr_cdf1,x)

function sparse_int(f::Function, a::Number, b::Number)
	#= Implements sparse grid quadrature from sparsegrids.de
	This implements the 1 dimensional rule that is exact for
	polynomials up to order 25.

	f = one dimensional function
	a = lower bound of integration
	b = upperbound of integration

	=#
	 weights = big([.0001816, .00063259999999999998, .0012895000000000001, .0021088000000000001, .0030577999999999998, .0041114999999999997, .0052490999999999996, .0064519, .0077034, .0089893000000000004, .0102971, .0116157, .0129348, .0142449, .0155368, .016801900000000002, .018032200000000002, .019219900000000002, .020357799999999999, .021439, .022457299999999999, .023406799999999998, .0242822, .0250786, .025791600000000001, .0264175, .0269527, .027394600000000002, .0277407, .027989199999999999, .028138799999999999, .0281888, .028138799999999999, .027989199999999999, .0277407, .027394600000000002, .0269527, .0264175, .025791600000000001, .0250786, .0242822, .023406799999999998, .022457299999999999, .021439, .020357799999999999, .019219900000000002, .018032200000000002, .016801900000000002, .0155368, .0142449, .0129348, .0116157, .0102971, .0089893000000000004, .0077034, .0064519, .0052490999999999996, .0041114999999999997, .0030577999999999998, .0021088000000000001, .0012895000000000001, .00063259999999999998, .0001816])

	nodes = big([.00006349999999999412, .00045090000000003183, .0013969000000000342, .0030839999999999756, .0056576000000000404, .009234399999999976, .013908600000000049, .01975439999999995, .02682859999999998, .035172599999999998, .044814400000000032, .055770399999999998, .068046000000000051, .08163699999999996, .096529700000000052, .11270170000000002, .13012199999999996, .14875190000000005, .16854519999999995, .18944850000000002, .21140210000000004, .23434010000000005, .25819099999999995, .28287810000000002, .30832029999999999, .33443230000000002, .36112509999999998, .3883067, .41588230000000004, .44375549999999997, .47182789999999997, .5, .52817210000000003, .55624450000000003, .58411769999999996, .6116933, .63887490000000002, .66556769999999998, .69167970000000001, .71712189999999998, .74180900000000005, .76565989999999995, .78859789999999996, .81055149999999998, .83145480000000005, .85124809999999995, .86987800000000004, .88729829999999998, .90347029999999995, .91836300000000004, .93195399999999995, .9442296, .95518559999999997, .9648274, .97317140000000002, .98024560000000005, .98609139999999995, .99076560000000002, .99434239999999996, .99691600000000002, .99860309999999997, .99954909999999997, .99993650000000001])

	f_evals = [f((b-a)*u + a) for u in nodes]
	return dot(f_evals, weights)*(b-a)
end


# defining limits of type distribution. Should be consistent with whatever PDF is used
lambda_lb = 0
lambda_ub = 1

max_mc = 20


#markets = convert(Vector
markets = [178]

for market in markets
	#products = levels(df[df[:mkt] .== market
	products = [650]
	
	for product in products
		# defning params for wholesaler
		#pdf(x) = weibull_pdf(x,1,0,1)
		#pdf(x) = unif_pdf(x)
		pdf(x) = burr_pdf(x)
		c = 10
		println("Working with Market: $market, Product: $product")

		# defining selection boolians so that we can select appropriate products/markets
		prod_bool = ((df[:product] .== product) & (df[:mkt] .== market))
		other_bool = ((df[:product] .!= product) & (df[:mkt] .== market))

		prod_chars = convert(Array,df[prod_bool,char_list])
		other_chars = convert(Array,df[other_bool,char_list])

		# flag that determines if product has matched price data
		matched_upc = (df[prod_bool, :_merge_purchases][1] == 3)
		M = df[prod_bool,:M][1]

		if matched_upc == true # Only need to work with matched products. No price sched for non-matched.
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

			#Defining share function which is ONLY a function of price
			function share(p)
				#= Function calculates shares of a given  product in a given market 
				based on price. Characteristics of the product and all other products
				are assumed fixed. 

				p = price =#
				
				num = exp(alpha*p + xb_prod)
				denom = 1 + num + sum(exp(xb_others))
				
				s = num/denom	

				return s
			end

			#Derivatives of the share fuction
			d_share(p) = ForwardDiff.derivative(share, p)
			dd_share(p) = ForwardDiff.derivative(d_share,p)
	
			#Defining p-star function. Requires solving NL equation
			function p_star(rho::Number,lambda::Number)
				g(p) =  (p - rho - lambda*max_mc)*d_share(p) + share(p)
				res = fzero(g,(rho+lambda*max_mc)) # upper bound here can be finicky
				return res
			end

			# interpolating p_star to avoid the messy root finding during solution
			p_star_grid = [p_star(rho,lambda) for rho = 1:100, lambda = 1:100]
			p_star_grid = convert(Array{Float64,2},p_star_grid)
			p_star_itp = interpolate(p_star_grid,BSpline(Quadratic(Line())),OnGrid())
		
			# Derivative of the p_star function. Need to use central difference approx
			# because auto diff doesn't with with fzero
			eps = 1e-16
			function d_pstar_d_rho(rho,lambda)
				res = (p_star(rho + eps,lambda) - p_star(rho - eps,lambda)) / (2*eps)
				return res
			end
			d_pstar_d_rho(rho,lambda) = 1

			#=function d_pstar_d_rho(rho,lambda)
				res = d_share(p_star(rho,lambda)) / (dd_share(p_star(rho,lambda))*(p_star(rho,lambda) - rho - lambda*max_mc) + 2*d_share(p_star(rho,lambda)))
				return res
			end=#

			
			# *****CODE THAT USES NLSOLVE*****

			# Defining Wholesaler FOCs which can then be solved. Following the syntax to use
			# NLsovle package.
			# NOTE: Can refomulate as constrained minimization if this doesn't work

			function wfocs!(theta::Vector, wfocs_vec)
				tic()
				nfocs = length(theta) # number of FOCs (last param is wholesaler cost)
				n = round(Int,(nfocs + 1)/2) # figuring out how many parts the price schedule has
				#println("Solving for $n part price schedule")
								
				lambda_vec = [lambda_lb ; theta[n+1:end] ; lambda_ub]
				rho_vec = theta[1:n]
				
				for i in 1:length(rho_vec)
					f(l) =  ((rho_vec[i] - c)*d_share(p_star(rho_vec[i],l))*d_pstar_d_rho(rho_vec[i],l) + share(p_star(rho_vec[i],l)))*pdf(l)
					wfocs_vec[i] = quadgk(f,big(lambda_vec[i]), big(lambda_vec[i+1]); abstol= 1e-16)[1]
				end

				for i in 1:(length(lambda_vec)-2) # indexing is inclusive so x:x = x for any x
					wfocs_vec[i + length(rho_vec)] = share(p_star(rho_vec[i],lambda_vec[i+1]))*(p_star(rho_vec[i],lambda_vec[i+1]) - lambda_vec[i+1]*max_mc - c) - share(p_star(rho_vec[i+1], lambda_vec[i+1]))*(p_star(rho_vec[i+1],lambda_vec[i+1]) - rho_vec[i+1] - lambda_vec[i+1]*max_mc + rho_vec[i] - c)
				end
				toc()
			end 
			wfocs_test = [0.0;0.0;0.0]
			x0 = [10.0;1.0;.50]
			wfocs!(x0,wfocs_test )
			println(wfocs_test)
			solution = nlsolve(wfocs!, x0,method=:trust_region, show_trace = true, ftol = 1e-12)
			println(solution)

			# Code using optimizer ****************
			#=
			opt = Opt(:LN_COBYLA,3)
			lower_bounds!(opt,[0,0,0])
			function objf(x,grad)
				if length(grad) > 0
				end
				return 1
			end
			min_objective!(opt,objf)
			equality_constraint!(opt,(res,x,g) -> wfocs!(res,x,g), [1e-6,1e-6,1e-6])
			
			(minf, minx, ret) = optimize(opt, [1.0, 11.0, 2.0])
			println(minx)	=#
	


	

	
		else
			println("Product has no matching price data.")
		end

	end
end
