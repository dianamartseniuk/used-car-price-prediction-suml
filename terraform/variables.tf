variable "location" {
  description = "The Azure region where resources will be created"
  type        = string
  default     = "polandcentral"
}

variable "resource_group_name" {
  description = "The name of the resource group"
  type        = string
  default     = "rg-usedcarprice-dev-we"
}

variable "acr_name" {
  description = "The name of the Azure Container Registry"
  type        = string
  default     = "acrusedcarpricedevpl"
}

variable "container_app_name" {
  description = "The name of the Container App"
  type        = string
  default     = "ca-usedcarprice-dev-pl"
}

variable "container_app_environment_name" {
  description = "The name of the Container App Environment"
  type        = string
  default     = "cae-loanapp-dev-plc"
}
