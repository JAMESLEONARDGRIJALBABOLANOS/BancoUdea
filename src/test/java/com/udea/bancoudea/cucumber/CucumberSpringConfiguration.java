package com.udea.bancoudea.cucumber;

import io.cucumber.spring.CucumberContextConfiguration;
import org.springframework.boot.test.context.SpringBootTest;

@CucumberContextConfiguration
@SpringBootTest
public class CucumberSpringConfiguration {
    // Configures Spring context for Cucumber tests — no additional setup required
}
