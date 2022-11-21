from .Place import Place
from .Calendar import Calendar
from .Parameters import Parameters

from typing import Dict
from repast4py.space import DiscretePoint as dpt

from .InsuranceProvider import shopInsuranceProviders

class Household(Place):
    def __init__(self, initDict: Dict):
        placeId = initDict['sp_id']
        location = dpt(x=int(initDict['x']), y=int(initDict['y']), z=0)
        super().__init__(placeId, location)

        self.hasInsurance = initDict['has_hazard_insurance'] == '1'
        self.isOwner = initDict['occupancy_status'] == 'owner_occupied'
        self.hasMortgage = initDict['owner_costs_with_mortgage'] != 'not_applicable'
        
        self.insurancePurchaseData = int(initDict['ins_purchase_date']) if self.hasInsurance else -1

        #### Bianica Additions
        self.fuelReductionLevel = 'none' ## Need to replace with actual current level of fuel reduction from properties file
        self.belowPoverty = initDict['below_poverty'] == 'TRUE'
        self.reasonNoInsurance = 'denied' ## Need to replace with reason household may not have insurance (denied,unaffordable,did_not_shop)
        self.structureLossProbability = 0.89 ## Need to replace with properties structure loss probability from properties file
        ####


    
    def step(self, calendar: Calendar, rng):
        if len(self.peopleAtPlace) != 0:
            self.perceivedRisk = self.peopleAtPlace[0].risk
        if calendar.isNewMonth:
            self.shopForInsurance(rng)
            self.reduceFuel()

    def shopForInsurance(self, rng):
        if self.hasInsurance:
            return
        if not self.isOwner:
            return
        if self.hasMortgage:
            shopInsuranceProviders(self)
            return

        pShop = rng.random()
        if self.perceivedRisk < Parameters.percievedRiskL:
            if pShop < Parameters.shopPL:
                shopInsuranceProviders(self)
        elif self.perceivedRisk < Parameters.perceivedRiskM:
            if pShop < Parameters.shopPM:
                shopInsuranceProviders(self)
        elif self.perceivedRisk < Parameters.perceivedRiskH:
            if pShop < Parameters.shopPH:
                shopInsuranceProviders(self)
    
    def reduceFuel(self):
        #### Bianica Additions
        if not self.isOwner:
            return
        if self.fuelReductionLevel == "full":
            return

        pReduce = rng.random()
        match self.reasonNoInsurance:
            case 'denied':
                if self.hasInsurance:
                    return
                if self.belowPoverty:
                    if pReduce < Parameters.fuelPL:
                        reduceFuelAtProperty(self)
                else
                    if self.hasMortgage:
                        if pReduce < Parameters.fuelPH:
                            reduceFuelAtProperty(self)
                    else
                        if self.perceivedRisk < Parameters.percievedRiskL:
                            if pReduce < Parameters.fuelPM:
                                reduceFuelAtProperty(self)
                        elif self.perceivedRisk < Parameters.perceivedRiskM:
                            if self.fuelReductionLevel == 'heavy':
                                if pReduce < Parameters.fuelPM:
                                   reduceFuelAtProperty(self)
                            else
                                if pReduce < Parameters.fuelPH:
                                    reduceFuelAtProperty(self)
                        elif self.perceivedRisk < Parameters.perceivedRiskH:
                            if pReduce < Parameters.fuelPH:
                                reduceFuelAtProperty(self)
            case 'did_not_shop':
                if self.hasInsurance:
                    return
                if self.hasMortgage:
                    return
                if self.belowPoverty:
                    return
                if self.perceivedRisk < Parameters.percievedRiskL:
                    if pReduce < Parameters.fuelPL:
                        reduceFuelAtProperty(self)
                elif self.perceivedRisk < Parameters.percievedRiskM:
                    if self.fuelReductionLevel == 'heavy':
                        if pReduce < Parameters.fuelPL:
                            reduceFuelAtProperty(self)
                    else
                        if pReduce < Parameters.fuelPM:
                            reduceFuelAtProperty(self)
                elif self.perceivedRisk < Parameters.percievedRiskH:
                    if self.fuelReductionLevel == 'heavy':
                        if pReduce < Parameters.fuelPM:
                            reduceFuelAtProperty(self)
                    else
                        if pReduce < Parameters.fuelPH:
                            reduceFuelAtProperty(self)
            case 'unaffordable':
                if self.belowPoverty:
                    if pReduce < Parameters.fuelPL:
                        reduceFuelAtProperty(self)
                else
                    if self.hasMortgage:
                        if pReduce < Parameters.fuelPH:
                            reduceFuelAtProperty(self)
                    else
                        if self.perceivedRisk < Parameters.percievedRiskL:
                            if pReduce < Parameters.fuelPL:
                                reduceFuelAtProperty(self)
                        elif self.perceivedRisk < Parameters.percievedRiskM:
                            if self.fuelReductionLevel == 'heavy':
                                if pReduce < Parameters.fuelPM:
                                    reduceFuelAtProperty(self)
                            else
                                if pReduce < Parameters.fuelPH:
                                    reduceFuelAtProperty(self)
                        elif self.perceivedRisk < Parameters.percievedRiskH:
                            if pReduce < Parameters.fuelPH:
                                reduceFuelAtProperty(self)
                case 'na': # household has insurance
                    if self.belowPoverty:
                        return
                    if self.perceivedRisk < Parameters.percievedRiskL:
                        reduceFuelAtProperty(self)
                    elif self.perceivedRisk < Parameters.perceivedRiskM 
                        if self.fuelReductionLevel == 'heavy':
                            if pReduce < Parameters.fuelPL:
                                reduceFuelAtProperty(self)
                        else 
                            if pReduce < Parameters.fuelPM:
                                reduceFuelAtProperty(self)
                    elif self.perceivedRisk < Parameters.percievedRiskH:
                        if pReduce < Parameters.fuelPH:
                            reduceFuelAtProperty(self)
        ####
        
    #### Bianica Additions
    def reduceFuelAtProperty(self):
        # update fuel reduction level of home
        match self.fuelReductionLevel:
            case 'none':
                self.fuelReductionLevel = 'light'
            case 'light':
                self.fuelReductionLevel = 'heavy'
            case 'heavy':
                self.fuelReductionLevel = 'full'
        
        # update probability of structure loss given the increased level of fuel reduction
        match self.fuelReductionLevel:
            case 'light':
                if self.structureLossProbability > 0.89:
                    self.structureLossProbability = 0.89
            case 'heavy':
                self.structureLossProbability = 0.55
            case 'full':
                self.structureLossProbability = 0.89
    ####

    def purchaseInsurance(self, offers):
        pass
