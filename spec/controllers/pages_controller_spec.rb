require 'rails_helper'

RSpec.describe PagesController, type: :controller do

  describe "GET #about" do
    it "returns http success" do
      get :about
      expect(response).to have_http_status(:success)
    end
  end

  describe "GET #blog" do
    it "returns http success" do
      get :blog
      expect(response).to have_http_status(:success)
    end
  end

  describe "GET #compare" do
    it "returns http success" do
      get :compare
      expect(response).to have_http_status(:success)
    end
  end

  describe "GET #contact" do
    it "returns http success" do
      get :contact
      expect(response).to have_http_status(:success)
    end
  end

  describe "GET #whatif" do
    it "returns http success" do
      get :whatif
      expect(response).to have_http_status(:success)
    end
  end

  describe "GET #backtests" do
    it "returns http success" do
      get :backtests
      expect(response).to have_http_status(:success)
    end
  end

end
