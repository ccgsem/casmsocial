from repast4py.space import DiscretePoint as dpt
from scipy.stats import triang, norm


class InsuranceProvider(object):
    def __init__(self, insurerId: int):
        self.id = insurerId
        self.riskTolerance = "low" # low, med, high
        self.threshold = 100 # arbitrarily assigning a threshold of 100 homes in a census block to each insurance provider (ie. rational is you don't want to insure all homes in a block because if it burns down it is catastrophic)
    
    def compute_premium(self, crps: float, bp: float, structVal: float, structYearBuilt: int):
        # crps: conditional risk to potential structures (ie. if there is a fire here and a home here, what is the chance it is burned down?)
        # bp: burn probability (ie. what is the chance a fire will be here)
        fireRisk = crps * bp

        # Adjust the fireRisk for the insurer's risk tolerance
        # Again, would love some evidence for the adjustment factor
        adjustmentFactor = norm.rvs(loc=.0001, scale=.00001, size=1, random_state=1)
        if self.riskTolerance == "low":
            fireRisk = fireRisk + adjustmentFactor
        elif self.riskTolerance == "high":
            fireRisk = fireRisk - adjustmentFactor

        # Premium calculation
        # Since there is a $200,000 dwelling replacement cost limit and $160,000 limit on the contents, I am creating a "replacmentCost" variable to capture the total potential payout from a policy

        # Structure

        # The US Army Corps of Engineers estimates that the replacement value of a structure depreciates by 1% each year for the first 20 years after it is built/renovated.
        # After 20 years, it is assumed that regular maintenance will keep the replacement value at 80% of their original value.
        # I am using the "EffectiveYear" column as the year when the 20 year counter should stop.
        replacementCostStructure = 0
        depreciationPercentage = 2022 - structYearBuilt
        if depreciationPercentage > 20:
            depreciationPercentage = 20
        depreciationPercentage = depreciationPercentage / 100

        # adjust the building value based on the depreciation precentage
        structAdjVal = structVal * (1 - depreciationPercentage)
        if structAdjVal > 200000:
            replacementCostStructure = 200000
        else:
            replacementCostStructure = structAdjVal

        # Contents

        # For the contents, the US Army Corps of Engineers estimates that the content-to-structure value ratio is .5 for all residential homes
        # This means that the estimted value of the contents in each residential home is half the value of the structure
        replacementCostContents = 0
        if structAdjVal * .5 > 160000:
            replacementCostContents = 160000
        else:
            replacementCostContents = structAdjVal * .5

        # Army Corps of Engineers NSI Technical Documentation - Structure Valuation and Occupancy Type
        # https://www.hec.usace.army.mil/confluence/nsi/technicalreferences/latest/technical-documentation#id-.TechnicalDocumentationv2022-StructureValuation

        # Calculate the premium value
        # I am creating a random variable given the aforementioned distribution around each fire risk point estimate
        # I would like some real substance/evidence behind the .0015 standard deviation number, but I don't have that
        assessedRisk = norm.rvs(loc=fireRisk, scale=.0015, size=1, random_state=1) # Create the risk distribution
        # risk_series = [x if x >=0 else 0 for x in dist]
        # cost_series =  [x * (replacementCostContents + replacementCostStructure) if x >= 0 else 0 for x in dist] 
        if assessedRisk < 0:
            assessedRisk = 0
        premium = assessedRisk * (replacementCostContents + replacementCostStructure)

        # Scale the premium up to model the full population
        

        return premium

    def provide_insurance(self, location: dpt, fireRisk: float):

        # Measure if the insurance provider already has too many homes in the census block to insure one more
        # number_of_homes_in_block = location.count_number_of_homes()
        # if number_of_homes_in_block > self.threshold:
        #   return False, _
        # else:

        # 
        premium = self.compute_premium()
        return provideInsurance, premium
        

providerMap = {}

def shopInsuranceProviders(hh):
    pass