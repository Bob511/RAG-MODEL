package java.com.example.demo;

import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class BasicController {

    @PostMapping("/auth/chat")
    public String login() {
        return "ChatAI";
    }
}