source "https://rubygems.org"
ruby "3.4.5"

# Ruby 3.4+ required gems (removed from standard library)
gem "csv"
gem "base64"
gem "bigdecimal"
gem "rake"
gem "uri"

# Core Jekyll with Ruby 3.4 compatibility - using patched versions
gem "jekyll", "~> 4.3"
gem "jekyll-sass-converter", "~> 2.2.0"
gem "faraday-retry"

# Sass - using sassc for better Ruby 3.4 compatibility
gem "sassc", "~> 2.4.0"

# Web server for local development
gem "webrick"

group :jekyll_plugins do
  gem "jekyll-paginate-v2"
  # gem "jekyll-autoprefixer"  # Temporarily disabled for Ruby 3.4 compatibility
  gem "jekyll-github-metadata"
  gem "jekyll-feed"
  gem "jekyll-sitemap"
  gem "jekyll-redirect-from"
  gem "jekyll-seo-tag"
  gem 'tzinfo'
  gem 'tzinfo-data'
end
