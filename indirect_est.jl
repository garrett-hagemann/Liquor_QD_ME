using DataArrays, DataFrames, ForwardDiff, NLsolve, Roots

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
	return 1
end


# defining limits of type distribution. Should be consistent with whatever PDF is used
lambda_lb = 0
lambda_ub = Inf

max_mc = 30


#markets = convert(Vector, levels(df[:,:mkt])) # needs to be a vector to play nice with rdiff
markets = [178]

for market in markets
	#products = levels(df[df[:mkt] .== market,:product])
	products = [650]
	
	for product in products
		# defning params for wholesaler
		pdf(x) = weibull_pdf(x,1,0,1)
		c = 7
		println("Working with Market: $market, Product: $product")

		# defining selection boolians so that we can select appropriate products/markets
		prod_bool = ((df[:product] .== product) & (df[:mkt] .== market))
		other_bool = ((df[:product] .!= product) & (df[:mkt] .== market))

		prod_chars = convert(Array,df[prod_bool,char_list])
		other_chars = convert(Array,df[other_bool,char_list])

		# flag that determines if product has matched price data
		matched_upc = (df[prod_bool, :_merge_purchases][1] == 3)

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
			function p_star(rho,lambda)
				g(p) =  (p - rho - lambda)*d_share(p) + share(p)
				res = fzero(g,5.0,order=8) # upper bound here can be finicky
				return res
			end
		
			# Derivative of the p_star function. Need to use central difference approx
			# because auto diff doesn't with with fzero
			eps = 1e-12
			function d_pstar_d_rho(rho,lambda)
				res = (p_star(rho + eps,lambda) - p_star(rho - eps,lambda)) / (2*eps)
				return res
			end
			d_pstar_d_rho(rho,lambda) = 1

			#=function d_pstar_d_rho(rho,lambda)
				res = d_share(p_star(rho,lambda)) / (dd_share(p_star(rho,lambda))*(p_star(rho,lambda) - rho - lambda*max_mc) + 2*d_share(p_star(rho,lambda)))
				return res
			end=#

			# Defining Wholesaler FOCs which can then be solved. Following the syntax to use
			# NLsovle package.
			# NOTE: Can refomulate as constrained minimization if this doesn't work

			function wfocs!(theta::Vector,wfocs_vec)
				tic()
				nfocs = length(theta) # number of FOCs (last param is wholesaler cost)
				n = round(Int,(nfocs + 1)/2) # figuring out how many parts the price schedule has
				#println("Solving for $n part price schedule")
								
				lambda_vec = [lambda_lb ; theta[n+1:end] ; lambda_ub]
				rho_vec = theta[1:n]
				
				for i in 1:length(rho_vec)
					f(l) =  ((rho_vec[i] - c)*d_share(p_star(rho_vec[i],l))*d_pstar_d_rho(rho_vec[i],l) + share(p_star(rho_vec[i],l)))*pdf(l)
					wfocs_vec[i] = quadgk(f,lambda_vec[i], lambda_vec[i+1])[1]
				end

				for i in 1:(nfocs-2) # indexing is inclusive so x:x = x for any x
					wfocs_vec[i + length(rho_vec)] = share(p_star(rho_vec[i],lambda_vec[i+1]))*(p_star(rho_vec[i],lambda_vec[i+1]) - lambda_vec[i+1] - c) - share(p_star(rho_vec[i+1], lambda_vec[i+1]))*(p_star(rho_vec[i+1],lambda_vec[i+1]) - rho_vec[i+1] - lambda_vec[i+1] + rho_vec[i] - c)
				end
				toc()
			end 
			println(p_star(10.0,6.0))	
			test = [0.0;9.0;0.0]
			wfocs!([30.0;11.0;2.0],test)
			println(test)
			#x0 = [40.0,16.0,.250] # "random" starting points
			#test = nlsolve(wfocs!,x0,method = :trust_region, show_trace = true, ftol = 1e-12)
			#println(test)

	

	
		else
			println("Product has no matching price data.")
		end

	end
end

