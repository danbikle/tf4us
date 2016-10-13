Rails.application.routes.draw do
  get  'pages/about'
  get  'pages/blog'
  get  'pages/compare'
  get  'pages/contact'
  get  'pages/whatif'
  root 'pages#home'
  get  'pages/backtests'
  
  # For details on the DSL available within this file, see http://guides.rubyonrails.org/routing.html
end
