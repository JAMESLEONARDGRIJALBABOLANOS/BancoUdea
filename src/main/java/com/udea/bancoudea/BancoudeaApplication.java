package com.udea.bancoudea;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication(scanBasePackages = "com.udea.bancoudea")
public class BancoudeaApplication {

	public static void main(String[] args) {
		SpringApplication.run(BancoudeaApplication.class, args);
	}

}
